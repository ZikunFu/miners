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

# Define additional helper functions
def distinct_n_grams(text, n):
    n_grams = set(zip(*[text[i:] for i in range(n)]))
    return len(n_grams) / len(text) if len(text) > 0 else 0

def process_model_output(output, num_tokens):
    pred_labels = output.strip().split()
    if len(pred_labels) < num_tokens:
        pred_labels.extend(['O'] * (num_tokens - len(pred_labels)))
    elif len(pred_labels) > num_tokens:
        pred_labels = pred_labels[:num_tokens]
    return pred_labels

# Evaluation Metrics Function
def evaluate_generation_metrics(hyps, refs):
    distinct_1_scores = []
    distinct_2_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE and Distinct metrics
    for hyp, ref in zip(hyps, refs):
        rouge_scores = rouge.score(ref, hyp)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        distinct_1 = distinct_n_grams(hyp.split(), 1)
        distinct_2 = distinct_n_grams(hyp.split(), 2)
        distinct_1_scores.append(distinct_1)
        distinct_2_scores.append(distinct_2)

    # Calculate BERTScore
    print("Calculating BERTScore...")
    P, R, F1 = bert_score(hyps, refs, lang='en', rescale_with_baseline=False)
    print("BERTScore calculated.")

    # Convert tensors to Python floats
    P, R, F1 = P.cpu().numpy().astype(float), R.cpu().numpy().astype(float), F1.cpu().numpy().astype(float)

    # Organize report dictionary
    report_dict = {
        "ROUGE-1": float(np.mean(rouge1_scores)),
        "ROUGE-2": float(np.mean(rouge2_scores)),
        "ROUGE-L": float(np.mean(rougeL_scores)),
        "BERTScore (P)": float(np.mean(P)),
        "BERTScore (R)": float(np.mean(R)),
        "BERTScore (F1)": float(np.mean(F1)),
        "Distinct-1": float(np.mean(distinct_1_scores)),
        "Distinct-2": float(np.mean(distinct_2_scores)),
        "Ensemble": float(np.mean([np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores), np.mean(F1)]))
    }

    return report_dict

# Main function setup
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="sentence-transformers/LaBSE", type=str, help="Path to embedding model")
    parser.add_argument("--gen_model_checkpoint", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help="Generation model")
    parser.add_argument("--dataset", type=str, default="masakhaner", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when available")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load embedding and generation models
    embedding_model = SentenceTransformer(args.model_checkpoint).cuda() if args.cuda else SentenceTransformer(args.model_checkpoint)
    gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint).cuda() if args.cuda else AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint)

    # Placeholder dataset - replace this section with actual data loading
    hyps = ["The quick brown fox jumps over the lazy dog", "A journey of a thousand miles begins with a single step"]
    refs = ["The quick brown fox leaped over a lazy dog", "A journey of a thousand miles starts with one step"]

    # Evaluate and display metrics
    print("Evaluating generation metrics...")
    report = evaluate_generation_metrics(hyps, refs)
    print("Evaluation Report:")
    print(json.dumps(report, indent=4))

    # Save report to JSON file
    output_dir = "logs/save_icl_NER"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "evaluation_report.json")
    with open(output_path, "w") as outfile:
        json.dump(report, outfile, indent=4)
    print(f"Report saved successfully to {output_path}")
