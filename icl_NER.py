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
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from seqeval.metrics import classification_report, f1_score


OPENAI_TOKEN = ""
COHERE_TOKEN = ""
HF_TOKEN = ""

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
    
def get_llama3_instruct_chat_response(gen_model, tokenizer, gen_model_checkpoint, messages, seed):
    print("messages: ", messages)
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
        pad_token_id=tokenizer.eos_token,
        max_new_tokens=64,  
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )

    response = outputs[0][input_ids.shape[-1]:]
    print("response: ", tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def retrieve_ids(train_embeddings, test_embeddings, train_labels, k, balance=False, all_possible_labels=[]):
    all_samples = []
    for test_id in tqdm(range(len(test_embeddings))):
        dists = []
        batch_size = 128
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
        "content": "You are a helpful assistant that performs named entity recognition (NER)."
    }
    messages.append(system_message)
    
    # Add few-shot examples as assistant and user turns
    for tokens, ner_tags in few_shot_examples:
        # User provides a sentence
        user_message = {
            "role": "user",
            "content": "Study this taxonomy for classifying named entities:- LOC (Location or physical facilities)- ORG (Organizations, corporations or other entities)- PER (Names of people)- DATE (Date or time)Identify all named entities in the following tokens:{" + " ".join(tokens)+"}\nAdditionally, you should add B- to the first token of a given entity and I- to subsequent ones if they exist. For tokens that are not named entities, mark them as O."
        }
        messages.append(user_message)
        # Assistant provides the tags
        assistant_message = {
            "role": "assistant",
            "content": " ".join(dataset.convert_ner_tags(ner_tags, to_labels=True))
        }
        messages.append(assistant_message)
    
    # Add the test sentence
    user_message = {
        "role": "user",
        "content": "Study this taxonomy for classifying named entities:- LOC (Location or physical facilities)- ORG (Organizations, corporations or other entities)- PER (Names of people)- DATE (Date or time)Identify all named entities in the following tokens:{" + " ".join(tokens)+"}\nAdditionally, you should add B- to the first token of a given entity and I- to subsequent ones if they exist. For tokens that are not named entities, mark them as O."
    }
    messages.append(user_message)
    return messages

def process_model_output(output, num_tokens):
    pred_labels = output.strip().split()
    # Handle mismatch in the number of tokens and predicted labels
    if len(pred_labels) < num_tokens:
        # Pad with 'O'
        pred_labels.extend(['O'] * (num_tokens - len(pred_labels)))
    elif len(pred_labels) > num_tokens:
        # Truncate to match the number of tokens
        pred_labels = pred_labels[:num_tokens]
    return pred_labels

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
        help="Path to pre-trained generation model (Llama 3.1)")
    parser.add_argument("--dataset", type=str, default="masakhaner", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when available")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="Prompt")
    parser.add_argument("--instruction", type=str, default="", help="Instruction")
    parser.add_argument("--k", type=int, default=1, help="Number of few-shot examples")
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Output directory setup
    output_dir = "logs/save_icl_NER"
    if args.load_in_8bit:
        output_dir += "_8bit"

    # Load embedding model (e.g., SentenceTransformer)
    embedding_model = SentenceTransformer(args.model_checkpoint).cuda()
    batch_size = 128

    # Load generation model (Llama 3.1)
    if args.load_in_8bit:
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN,
            device_map="auto",
            load_in_8bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN,
            clean_up_tokenization_spaces=True
        )
    else:
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.gen_model_checkpoint,
            token=HF_TOKEN,
            clean_up_tokenization_spaces=True
        )

    # Load MasakhaNERDataset
    if args.dataset == "masakhaner":
        dataset = MasakhaNERDataset(prompt=args.prompt, sample_size=10)

    for lang in dataset.LANGS:
        print(f"Processing language: {lang}")

        # Get train and test data
        train_data = dataset.train_data
        test_data = dataset.test_data

        train_tokens = [sample['tokens'] for sample in train_data]
        train_tags = [sample['ner_tags'] for sample in train_data]
        test_tokens = [sample['tokens'] for sample in test_data]
        test_tags = [sample['ner_tags'] for sample in test_data]

        # Prepare texts for embedding
        train_texts = [" ".join(tokens) for tokens in train_tokens]
        test_texts = [" ".join(tokens) for tokens in test_tokens]

        # Compute embeddings
        print("Computing embeddings for training data...")
        train_embeddings = embedding_model.encode(train_texts, convert_to_numpy=True, show_progress_bar=True)
        print("Computing embeddings for test data...")
        test_embeddings = embedding_model.encode(test_texts, convert_to_numpy=True, show_progress_bar=True)

        # Retrieve k-nearest neighbors
        if args.k > 0:
            all_few_shot_samples_ids = retrieve_ids(
                train_embeddings, test_embeddings, train_tags, k=args.k
            )

        hyps = []
        prompts = []
        for text_id in tqdm(range(len(test_texts))):
            test_tokens_example = test_tokens[text_id]

            # Prepare few-shot examples
            few_shot_examples = []
            if args.k > 0:
                for few_shot_sample_id in all_few_shot_samples_ids[text_id]:
                    tokens = train_tokens[few_shot_sample_id]
                    labels = train_tags[few_shot_sample_id]
                    few_shot_examples.append((tokens, labels))

            # Construct prompt
            messages = construct_prompt(few_shot_examples, test_tokens_example)
            prompts.append(messages)
            
            # Get model output
            hyp = get_llama3_instruct_chat_response(
                gen_model, tokenizer, args.gen_model_checkpoint, messages, args.seed
            )

            # Process model output
            num_tokens = len(test_tokens_example)
            pred_labels = process_model_output(hyp, num_tokens)
            hyps.append(pred_labels)
            

        # Evaluate using seqeval
        f1 = f1_score(test_tags, hyps)
        report = classification_report(test_tags, hyps)
        print(f"F1 Score for {lang}: {f1}")
        print(f"Classification Report for {lang}:\n{report}")

        # Save results
        if not os.path.exists(f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/")

        obj = {'f1': f1, 'classification_report': report}
        preds = {"hyp": hyps, "gold": test_tags}
        all_prompts = {"prompts": prompts}

        file_path = f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/eval_{lang}_{args.k}.json"
        with open(file_path, "w") as outfile:
            json.dump(obj, outfile, indent=4)

        file_path = f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/eval_{lang}_{args.k}_preds.json"
        with open(file_path, "w") as outfile:
            json.dump(preds, outfile, indent=4)

        file_path = f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/eval_{lang}_{args.k}_prompts.json"
        with open(file_path, "w") as outfile:
            json.dump(all_prompts, outfile, indent=4)
        break
