import os
import torch
import random
import numpy as np
import argparse
import json
import hashlib
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer
from utils import MasakhaNERDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from seqeval.metrics import classification_report
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from transformers.utils import logging

logging.set_verbosity_error() 
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

OPENAI_TOKEN = ""
COHERE_TOKEN = ""
HF_TOKEN = "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"

def argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),16) % len(max_indices)
    return max_indices[idx]

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def normalize(x):
    x = np.array(x)
    return np.exp(x - logsumexp(x))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def get_llama3_instruct_chat_response(gen_model, tokenizer, gen_model_checkpoint, messages, seed,verbose=True):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(gen_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=64,  
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )
    inputs = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    if(verbose):
        print("\n"+"="*35+"INPUT"+"="*35)
        print(inputs)
        print("="*35+"RESPONSE"+"="*43)
        print(response)
        print("="*70)
    return response

def retrieve_ids(train_embeddings, test_embeddings, train_labels, k, balance=False, all_possible_labels=[]):
    all_samples = []
    for test_id in tqdm(range(len(test_embeddings))):
        dists = []
        batch_size = 1                              ########change back to 128 
        if len(train_embeddings) < batch_size:
            batch_size = len(test_embeddings) // 2
        
        num_of_batches = len(train_embeddings) // batch_size

        if (len(train_embeddings) % batch_size) > 0:
            num_of_batches += 1

        for i in range(num_of_batches):
            train_embedding = torch.FloatTensor(train_embeddings[i*batch_size:(i+1)*batch_size]).unsqueeze(1).cuda()
            
            test_embedding = torch.FloatTensor(test_embeddings[test_id]).unsqueeze(0)
            test_embedding = test_embedding.expand(len(train_embedding), -1).unsqueeze(1).cuda()
            
            dist = torch.cdist(test_embedding, train_embedding, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary').squeeze().tolist()

            if isinstance(dist, float):
                dist = [dist]

            for j in range(len(dist)):
                dists.append([dist[j], train_labels[i*batch_size + j], i*batch_size + j])

        if balance:
            sorted_dists = sorted(dists, key=lambda l: l[0], reverse=False)
        else:
            sorted_dists = sorted(dists, key=lambda l: l[0], reverse=False)[:k]

        all_indices = []
        if balance:
            for opt in all_possible_labels:
                count_found = 0
                for obj in sorted_dists:
                    if opt == obj[1]:
                        all_indices.append(obj[2])
                        count_found += 1
                        if count_found == k:
                            break
        else:
            all_indices = [obj[2] for obj in sorted_dists]
        all_samples.append(all_indices)
    return all_samples

def construct_prompt(few_shot_examples, test_tokens):
    messages = []
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that performs open ended generation and outputs completed prompts in multiple languages such as: english, french, afgan, german"
    }
                                                                         
    messages.append(system_message)
    
    for tokens, ner_tags in few_shot_examples:
        temp = f'''
        finish these prompts: 
        "once upon a time"
        "Describe a day in the life of a lighthouse keeper on a remote island."
        "What do you think the future of transportation will look like in 50 years?"
        "Imagine you are a traveler in ancient China. What sights and experiences would you have?"
        '''
        user_message = {
            "role": "user",
            "content": temp
        }
        messages.append(user_message)
        assistant_message = {
            "role": "assistant",
            "content": " ".join(dataset.convert_ner_tags(ner_tags, to_BIO=True))
        }
        messages.append(assistant_message)
    
    temp = f'''
         finish these prompts: 
        "once upon a time"
        "Describe a day in the life of a lighthouse keeper on a remote island."
        '''
    user_message = {
        "role": "user",
        "content": temp    
        }
    messages.append(user_message)
    return messages

def process_model_output(output, num_tokens):
    pred_labels = output.strip().split()
    if len(pred_labels) < num_tokens:
        pred_labels.extend(['O'] * (num_tokens - len(pred_labels)))
    elif len(pred_labels) > num_tokens:
        pred_labels = pred_labels[:num_tokens]
    return pred_labels

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def distinct_n_grams(text, n):
    n_grams = set(zip(*[text[i:] for i in range(n)]))
    return len(n_grams) / len(text) if len(text) > 0 else 0

def evaluate_generation_metrics(hyps, refs):
    distinct_1_scores = []
    distinct_2_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for hyp, ref in zip(hyps, refs):
        rouge_scores = rouge.score(ref, hyp)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        distinct_1 = distinct_n_grams(hyp.split(), 1)
        distinct_2 = distinct_n_grams(hyp.split(), 2)
        distinct_1_scores.append(distinct_1)
        distinct_2_scores.append(distinct_2)

    P, R, F1 = bert_score(hyps, refs, lang='en', rescale_with_baseline=True)

    report_dict = {
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-2": np.mean(rouge2_scores),
        "ROUGE-L": np.mean(rougeL_scores),
        "BERTScore (P)": np.mean(P),
        "BERTScore (R)": np.mean(R),
        "BERTScore (F1)": np.mean(F1),
        "Distinct-1": np.mean(distinct_1_scores),
        "Distinct-2": np.mean(distinct_2_scores),
        "Ensemble": np.mean([np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores), np.mean(F1)])
    }

    return report_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        default="sentence-transformers/LaBSE",
        type=str,
        help="Path to pre-trained embedding model")
    parser.add_argument(
        "--gen_model_checkpoint",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help="Path to pre-trained generation model")
    parser.add_argument("--dataset", type=str, default="masakhaner", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when available")
    parser.add_argument("--load_in_4bit",default=True, action="store_true", help="Load model in 4-bit precision") 
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="Prompt")
    parser.add_argument("--instruction", type=str, default="", help="Instruction")
    parser.add_argument("--k", type=int, default=2, help="Number of few-shot examples")
    args = parser.parse_args()

    print("###########################")
    print("dataset:", args.dataset)
    print("model_checkpoint:", args.model_checkpoint)
    print("gen_model_checkpoint:", args.gen_model_checkpoint)
    print("seed:", args.seed)
    print("k:", args.k)
    print("###########################")

    set_seed(args.seed)

    output_dir = "logs/save_icl_NER"
    if args.load_in_4bit:
        output_dir += "_4bit"

    embedding_model = SentenceTransformer(args.model_checkpoint).cuda()

    if args.load_in_4bit:
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN,
            device_map="auto",
            load_in_4bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN
        )
    else:
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN
        )

    if args.dataset == "masakhaner":
        dataset = MasakhaNERDataset()

    for lang in dataset.LANGS:
        print(f"Processing language: {lang}")

        train_data = dataset.train_data[lang]
        test_data = dataset.test_data[lang]

        train_tokens = [sample['tokens'] for sample in train_data]
        train_tags = [sample['ner_tags'] for sample in train_data]
        test_tokens = [sample['tokens'] for sample in test_data]
        test_tags = [sample['ner_tags'] for sample in test_data]
        test_tags_bio = [dataset.convert_ner_tags(tags, to_BIO=True) for tags in test_tags]

        train_texts = [" ".join(tokens) for tokens in train_tokens]
        test_texts = [" ".join(tokens) for tokens in test_tokens]
        
        print("Computing embeddings for training data...")
        train_embeddings = embedding_model.encode(train_texts, convert_to_numpy=True, show_progress_bar=True)
        print("Computing embeddings for test data...")
        test_embeddings = embedding_model.encode(test_texts, convert_to_numpy=True, show_progress_bar=True)

        if args.k > 0:
            all_few_shot_samples_ids = retrieve_ids(
                train_embeddings, test_embeddings, train_tags, k=args.k
            )

        hyps = []
        refs = [" ".join(tags) for tags in test_tags_bio]  
        prompts = []
        
        for text_id in tqdm(range(len(test_texts))):
            test_token = test_tokens[text_id]

            few_shot_examples = []
            if args.k > 0:
                for few_shot_sample_id in all_few_shot_samples_ids[text_id]:
                    tokens = train_tokens[few_shot_sample_id]
                    labels = train_tags[few_shot_sample_id]
                    few_shot_examples.append((tokens, labels))

            messages = construct_prompt(few_shot_examples, test_token)
            prompts.append(messages)
            
            hyp = get_llama3_instruct_chat_response(
                gen_model, tokenizer, args.gen_model_checkpoint, messages, args.seed
            )

            pred_labels = process_model_output(hyp, num_tokens=len(test_token))
            hyps.append(" ".join(pred_labels))  
        
        report = evaluate_generation_metrics(hyps, refs)
        print(f"Evaluation Report for {lang}:\n{report}")

        if not os.path.exists(f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/")

        preds = {"hyp": hyps, "gold": test_tags_bio}
        all_prompts = {"prompts": prompts}

        file_path = f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/eval_{lang}_{args.k}.json"
        with open(file_path, "w") as outfile:
            json.dump(report, outfile, indent=4)

        file_path = f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/eval_{lang}_{args.k}_preds.json"
        with open(file_path, "w") as outfile:
            json.dump(preds, outfile, indent=4)

        file_path = f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/eval_{lang}_{args.k}_prompts.json"
        with open(file_path, "w") as outfile:
            json.dump(all_prompts, outfile, indent=4)
