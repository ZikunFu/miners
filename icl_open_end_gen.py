import os
import time
import torch
import random
import numpy as np
import argparse
import json
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import XLSumDataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers.utils import logging
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')

logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

HF_TOKEN = "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"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_gemma_chat_response(gen_model, tokenizer, messages, verbose=True):
    # Extract only the content field from each message dictionary for tokenization
    text_inputs = [msg['content'] for msg in messages]

    input_ids = tokenizer(text_inputs, return_tensors="pt", padding=True).input_ids.to(gen_model.device)

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=600,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def construct_prompt(few_shot_examples, model_checkpoint, language):
    messages = []

    if model_checkpoint == "google/gemma-2-9b-it":
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant skilled in open-ended generation across multiple languages. "
                       "Please respond to each prompt in the appropriate language based on the language setting."
        }
        messages.append(system_message)

    open_ended_prompts = [
        "Describe a festival in a mystical world where every season is celebrated uniquely.",
        "Imagine a future where humans can communicate with animals. Describe a conversation between two unlikely friends.",
        "What would a day in the life of an astronaut exploring a new galaxy look like?",
        "Tell the story of an inventor who creates a device to see the memories of any object they touch.",
        "Describe a city of the future where nature and technology exist in perfect harmony."
    ]

    for prompt in open_ended_prompts:
        user_message = {
            "role": "user",
            "content": f"Here is a prompt: {prompt}, please continue the story {language}"
        }
        messages.append(user_message)

    return messages

def get_reference_texts():
    return [
        "In the mystical realm of Etherea, every season brings a grand festival...",
        "In this new world, a human and a fox share a curious bond...",
        "Every dawn in the Omega galaxy is a kaleidoscope of colors...",
        "Dr. Emara’s Memory Device became legendary...",
        "The city of NeoTerra is a lush blend of green spaces and tech marvels..."
    ]

def evaluate_generation_metrics(hyps, refs):
    distinct_1_scores = []
    distinct_2_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for hyp, ref in zip(hyps, refs):
        rouge_scores = rouge.score(ref, hyp)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        meteor_scores.append(meteor_score([ref.split()], hyp.split()))

        distinct_1 = len(set(zip(*[hyp.split()[i:] for i in range(1)]))) / len(hyp.split()) if len(hyp.split()) > 0 else 0
        distinct_2 = len(set(zip(*[hyp.split()[i:] for i in range(2)]))) / len(hyp.split()) if len(hyp.split()) > 1 else 0
        distinct_1_scores.append(distinct_1)
        distinct_2_scores.append(distinct_2)

    print("Calculating BERTScore with 8-bit precision...")
    P, R, F1 = bert_score(hyps, refs, lang='en', rescale_with_baseline=False, model_type="microsoft/deberta-xlarge-mnli", device='cuda', num_layers=12)
    print("BERTScore calculated.")

    P, R, F1 = P.cpu().numpy().astype(float), R.cpu().numpy().astype(float), F1.cpu().numpy().astype(float)

    report_dict = {
        "ROUGE-1": float(np.mean(rouge1_scores)),
        "ROUGE-2": float(np.mean(rouge2_scores)),
        "ROUGE-L": float(np.mean(rougeL_scores)),
        "METEOR": float(np.mean(meteor_scores)),
        "BERTScore (P)": float(np.mean(P)),
        "BERTScore (R)": float(np.mean(R)),
        "BERTScore (F1)": float(np.mean(F1)),
        "Distinct-1": float(np.mean(distinct_1_scores)),
        "Distinct-2": float(np.mean(distinct_2_scores)),
        "Ensemble": float(np.mean([np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores), np.mean(F1)]))
    }

    return report_dict

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Total Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        print("CUDA is not available.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="sentence-transformers/LaBSE", type=str)
    parser.add_argument("--gen_model_checkpoint", default="google/gemma-2-9b-it", type=str)
    parser.add_argument("--dataset", type=str, default="xlsum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args([])  # Adjusted for notebook execution

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = SentenceTransformer(args.model_checkpoint).to(device)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model_checkpoint,
        token=HF_TOKEN,
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)

    output_dir = "generated_responses"
    metrics_dir = os.path.join(output_dir, "metrics")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    dataset = XLSumDataset(sample_size=10)
    selected_languages = list(dataset.train_data.keys())
    all_metrics = []

    start_time = time.time()
    reference_texts = get_reference_texts()

    for k in [2, 10]:  # Few-shot examples loop
        for language in tqdm(selected_languages, desc="Processing Selected Languages", unit="language"):
            if language in dataset.train_data:
                messages = construct_prompt(few_shot_examples=[], model_checkpoint=args.gen_model_checkpoint, language=language)
                language_output = []

                for prompt_id, (message, ref) in enumerate(zip(messages, reference_texts), start=1):
                    response = get_gemma_chat_response(gen_model, tokenizer, [message], verbose=False)
                    hyps = [response]
                    refs = [ref]
                    
                    # Save individual prompt and response for each language in a text format
                    language_output.append(f"Prompt Number: {prompt_id}\nPrompt: {message['content']}\nResponse: {response}\n")

                    # Evaluate and collect metrics
                    prompt_metrics = evaluate_generation_metrics(hyps, refs)
                    prompt_metrics["Prompt Number"] = prompt_id
                    prompt_metrics["Language"] = language
                    prompt_metrics["k"] = k
                    all_metrics.append(prompt_metrics)

                    elapsed_time = time.time() - start_time
                    est_time_left = (len(messages) * len(selected_languages) - prompt_id) * (elapsed_time / (prompt_id + 1))
                    print(f"Estimated Time Remaining: {est_time_left:.2f} seconds", end='\r')

                # Save all prompts and responses for each language to a text file
                language_file = os.path.join(output_dir, f"{language}_prompts_k{k}.txt")
                with open(language_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(language_output))

    # Save all metrics in a single JSON file for structured analysis
    with open(os.path.join(metrics_dir, "multilingual_evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    total_time = time.time() - start_time
    print(f"\nAll responses generated and saved successfully in {total_time:.2f} seconds.")
