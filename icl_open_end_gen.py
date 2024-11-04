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
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import XLSumDataset  # Updated to use XLSumDataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers.utils import logging

logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

OPENAI_TOKEN = ""
COHERE_TOKEN = ""
HF_TOKEN = "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"

def argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(), 16) % len(max_indices)
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

def get_llama3_instruct_chat_response(gen_model, tokenizer, messages, verbose=True):
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
    if verbose:
        print("\n" + "="*35 + "INPUT" + "="*35)
        print(inputs)
        print("="*35 + "RESPONSE" + "="*43)
        print(response)
        print("="*70)
    return response

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

        distinct_1 = len(set(zip(*[hyp[i:] for i in range(1)]))) / len(hyp.split()) if len(hyp.split()) > 0 else 0
        distinct_2 = len(set(zip(*[hyp[i:] for i in range(2)]))) / len(hyp.split()) if len(hyp.split()) > 1 else 0
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="sentence-transformers/LaBSE", type=str, help="Path to embedding model")
    parser.add_argument("--gen_model_checkpoint", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help="Generation model")
    parser.add_argument("--dataset", type=str, default="xlsum", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when available")
    parser.add_argument("--sample_size", type=int, default=5, help="Limit the dataset sample size")
    args = parser.parse_args([])  # Adjusted for notebook execution

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load embedding and generation models
    embedding_model = SentenceTransformer(args.model_checkpoint).cuda() if args.cuda else SentenceTransformer(args.model_checkpoint)
    gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint).cuda() if args.cuda else AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint)

    # Load the XLSum dataset
    dataset = XLSumDataset(sample_size=args.sample_size)

    # Choose the correct language key and get sample data
    lang = "english"  # Adjusted to match available keys in XLSum
    if lang in dataset.train_data:
        train_texts = dataset.train_data[lang]["source"]
        train_summaries = dataset.train_data[lang]["target"]

        # Evaluate and display metrics for a sample
        hyps = train_texts[:10]
        refs = train_summaries[:10]

        # Calculate and display metrics
        print("Evaluating generation metrics...")
        report = evaluate_generation_metrics(hyps, refs)
        print("Evaluation Report:")
        print(json.dumps(report, indent=4))

        # Save report to JSON file
        output_dir = "logs/save_xlsum"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "evaluation_report.json")
        with open(output_path, "w") as outfile:
            json.dump(report, outfile, indent=4)
        print(f"Report saved successfully to {output_path}")
    else:
        print(f"Language '{lang}' not found in the dataset.")
