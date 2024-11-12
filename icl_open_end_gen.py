# import os
# import time  # Added for time estimation
# import torch
# import random
# import numpy as np
# import argparse
# import json
# import hashlib
# from tqdm import tqdm  # Import tqdm for loading bars
# from collections import Counter
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from utils import XLSumDataset  # Updated to use XLSumDataset
# from rouge_score import rouge_scorer
# from bert_score import score as bert_score
# from transformers.utils import logging
# from datasets import load_dataset
# from nltk.translate.meteor_score import meteor_score
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import nltk
# nltk.download('wordnet')

# logging.set_verbosity_error()
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# OPENAI_TOKEN = ""
# COHERE_TOKEN = ""
# HF_TOKEN = "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"

# def argmax(array):
#     """argmax with deterministic pseudorandom tie breaking."""
#     max_indices = np.arange(len(array))[array == np.max(array)]
#     idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(), 16) % len(max_indices)
#     return max_indices[idx]

# def logsumexp(x):
#     c = x.max()
#     return c + np.log(np.sum(np.exp(x - c)))

# def normalize(x):
#     x = np.array(x)
#     return np.exp(x - logsumexp(x))

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

# def get_gemma_instruct_chat_response(gen_model, tokenizer, messages, verbose=True):
#     # Extract the "content" field from each message
#     input_texts = [msg["content"] for msg in messages]

#     # Tokenize the extracted input texts
#     input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).input_ids.to(gen_model.device)

#     outputs = gen_model.generate(
#         input_ids,
#         max_new_tokens=600,
#         do_sample=True,
#         temperature=0.2,
#         top_p=1
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if verbose:
#         print("\n" + "="*35 + "RESPONSE" + "="*43)
#         print(response)
#         print("="*70)
#     return response

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if verbose:
#         print("\n" + "="*35 + "RESPONSE" + "="*43)
#         print(response)
#         print("="*70)
#     return response

# # Construct prompt for open-ended generation task across multiple languages
# def construct_prompt(few_shot_examples, model_checkpoint, language):
#     messages = []
#     assistant_role = "model"
    
#     # System message with multilingual instruction
#     if model_checkpoint == "google/gemma-2-9b-it":
#         system_message = {
#             "role": "system",
#             "content": "You are a helpful assistant skilled in open-ended generation across multiple languages. "
#                        "Please respond to each prompt in the appropriate language based on the language setting."
#         }
#         messages.append(system_message)
    
#     # Define open-ended prompts in the preferred format
#     open_ended_prompts = [
#         "Describe a festival in a mystical world where every season is celebrated uniquely.",
#         "Imagine a future where humans can communicate with animals. Describe a conversation between two unlikely friends.",
#         "What would a day in the life of an astronaut exploring a new galaxy look like?",
#         "Tell the story of an inventor who creates a device to see the memories of any object they touch.",
#         "Describe a city of the future where nature and technology exist in perfect harmony.",
#     ]
    
#     # Create user prompts for each language
#     for prompt in open_ended_prompts:
#         user_message = {
#             "role": "user",
#             "content": f"Here is a prompt: {prompt}, please continue the story {language}"
#         }
#         messages.append(user_message)
    
#     return messages

# def evaluate_generation_metrics(hyps, refs):
#     distinct_1_scores = []
#     distinct_2_scores = []
#     rouge1_scores = []
#     rouge2_scores = []
#     rougeL_scores = []
#     meteor_scores = []

#     rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     for hyp, ref in zip(hyps, refs):
#         rouge_scores = rouge.score(ref, hyp)
#         rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
#         rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
#         rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
#         meteor_scores.append(meteor_score([ref.split()], hyp.split()))

#         distinct_1 = len(set(zip(*[hyp.split()[i:] for i in range(1)]))) / len(hyp.split()) if len(hyp.split()) > 0 else 0
#         distinct_2 = len(set(zip(*[hyp.split()[i:] for i in range(2)]))) / len(hyp.split()) if len(hyp.split()) > 1 else 0
#         distinct_1_scores.append(distinct_1)
#         distinct_2_scores.append(distinct_2)

#     print("Calculating BERTScore...")
#     P, R, F1 = bert_score(hyps, refs, lang='en', rescale_with_baseline=False)
#     print("BERTScore calculated.")

#     P, R, F1 = P.cpu().numpy().astype(float), R.cpu().numpy().astype(float), F1.cpu().numpy().astype(float)

#     report_dict = {
#         "ROUGE-1": float(np.mean(rouge1_scores)),
#         "ROUGE-2": float(np.mean(rouge2_scores)),
#         "ROUGE-L": float(np.mean(rougeL_scores)),
#         "METEOR": float(np.mean(meteor_scores)),
#         "BERTScore (P)": float(np.mean(P)),
#         "BERTScore (R)": float(np.mean(R)),
#         "BERTScore (F1)": float(np.mean(F1)),
#         "Distinct-1": float(np.mean(distinct_1_scores)),
#         "Distinct-2": float(np.mean(distinct_2_scores)),
#         "Ensemble": float(np.mean([np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores), np.mean(F1)]))
#     }

#     return report_dict

# def evaluate_per_prompt_and_store(hyps, refs, prompt_id, language, metrics_dir):
#     report = evaluate_generation_metrics(hyps, refs)
#     language_dir = os.path.join(metrics_dir, language)
#     if not os.path.exists(language_dir):
#         os.makedirs(language_dir)
#     metrics_file = os.path.join(language_dir, f"prompt_{prompt_id}_metrics.json")
#     with open(metrics_file, "w", encoding="utf-8") as f:
#         json.dump(report, f, indent=4)
    
#     print(f"Metrics for prompt {prompt_id} in {language} saved to {metrics_file}")
#     return report

# if __name__ == "__main__":

#     if torch.cuda.is_available():
#         print("CUDA is available!")
#         print("Device Name:", torch.cuda.get_device_name(0))
#         print("Total Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
#     else:
#         print("CUDA is not available.")

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_checkpoint", default="sentence-transformers/LaBSE", type=str)
#     parser.add_argument("--gen_model_checkpoint", default="google/gemma-2-9b-it", type=str)  # Changed to Gemma
#     parser.add_argument("--dataset", type=str, default="xlsum")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--cuda", action="store_true")
#     args = parser.parse_args([])  # Adjusted for notebook execution

#     set_seed(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     embedding_model = SentenceTransformer(args.model_checkpoint).to(device)
    
#     # Load the model in 8-bit precision
#     gen_model = AutoModelForCausalLM.from_pretrained(
#         "google/gemma-2-9b-it",
#         load_in_8bit=True,
#         device_map="auto",
#         token=HF_TOKEN  # Updated to use `token` instead of `use_auth_token`
#     )
#     tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

#     #gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint, torch_dtype=torch.float16).to(device)  # Load Gemma with 16-bit
#     #tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint)

#     output_dir = "generated_responses"
#     metrics_dir = os.path.join(output_dir, "metrics")
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(metrics_dir, exist_ok=True)

#     selected_languages = ["english", "french", "spanish", "german", "chinese_simplified", "arabic", "hindi", "swahili", "russian", "japanese"]
#     all_metrics = []  
#     start_time = time.time()
#     dataset = XLSumDataset(sample_size=10)  # Adjust `sample_size` as needed

#     for language in tqdm(selected_languages, desc="Processing Selected Languages", unit="language"):
#         if language in dataset.train_data:
#             messages = construct_prompt(few_shot_examples=[], model_checkpoint=args.gen_model_checkpoint, language=language)
            
#             for prompt_id, message in enumerate(tqdm(messages, desc=f"Processing Prompts in {language}", unit="prompt", leave=False)):
#                 response = get_gemma_instruct_chat_response(gen_model, tokenizer, [message], verbose=False)
#                 hyps = [response]
#                 refs = ["Sample reference text"]
                
#                 prompt_metrics = evaluate_per_prompt_and_store(hyps, refs, prompt_id+1, language, metrics_dir)
#                 all_metrics.append(prompt_metrics)

#                 elapsed_time = time.time() - start_time
#                 est_time_left = (len(messages) * len(selected_languages) - prompt_id) * (elapsed_time / (prompt_id + 1))
#                 print(f"Estimated Time Remaining: {est_time_left:.2f} seconds", end='\r')

#     with open(os.path.join(metrics_dir, "multilingual_evaluation_report.json"), "w", encoding="utf-8") as f:
#         json.dump(all_metrics, f, indent=4)
#     total_time = time.time() - start_time
#     print(f"\nAll responses generated and saved successfully in {total_time:.2f} seconds.")




import os
import time  # Added for time estimation
import torch
import random
import numpy as np
import argparse
import json
import hashlib
from tqdm import tqdm  # Import tqdm for loading bars
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import XLSumDataset  # Updated to use XLSumDataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers.utils import logging
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt  # For plotting F1 scores
import nltk
nltk.download('wordnet')

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
        max_new_tokens=600,  
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

# Construct prompt for open-ended generation task across multiple languages
def construct_prompt(few_shot_examples, model_checkpoint, language):
    messages = []
    assistant_role = "assistant"
    
    # System message with multilingual instruction
    if model_checkpoint == "Meta-Llama-3.1-8B-Instruct":
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant skilled in open-ended generation across multiple languages. "
                       "Please respond to each prompt in the appropriate language based on the language setting. "
                       "The languages supported are amharic, arabic, azerbaijani, bengali, burmese, "
                       "chinese_simplified, chinese_traditional, english, french, gujarati, hausa, hindi, "
                       "igbo, indonesian, japanese, kirundi, korean, kyrgyz, marathi, nepali, oromo, pashto, persian, "
                       "pidgin, portuguese, punjabi, russian, scottish_gaelic, serbian_cyrillic, serbian_latin, "
                       "sinhala, somali, spanish, swahili, tamil, telugu, thai, tigrinya, turkish, ukrainian, urdu, "
                       "uzbek, vietnamese, welsh, yoruba."
        }
        messages.append(system_message)
    elif model_checkpoint == "google/gemma-2-9b-it":
        assistant_role = "model"
    
    # Define open-ended prompts in the preferred format
    open_ended_prompts = [
        "Describe a festival in a mystical world where every season is celebrated uniquely.",
        "Imagine a future where humans can communicate with animals. Describe a conversation between two unlikely friends.",
        "What would a day in the life of an astronaut exploring a new galaxy look like?",
        "Tell the story of an inventor who creates a device to see the memories of any object they touch.",
        "Describe a city of the future where nature and technology exist in perfect harmony.",
    ]
    
    # Create user prompts for each language and the 50 provided prompts
    for prompt in open_ended_prompts:
        user_message = {
            "role": "user",
            "content": f"Here is a prompt: {prompt}, please continue the story {language}"
        }
        messages.append(user_message)
    
    return messages

# Process model output to align with token count
def process_model_output(output, num_tokens):
    pred_labels = output.strip().split()
    if len(pred_labels) < num_tokens:
        pred_labels.extend(['O'] * (num_tokens - len(pred_labels)))
    elif len(pred_labels) > num_tokens:
        pred_labels = pred_labels[:num_tokens]
    return pred_labels

# Convert numpy types to native Python types for JSON serialization
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

    print("Calculating BERTScore...")
    P, R, F1 = bert_score(hyps, refs, lang='en', rescale_with_baseline=False)
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

def evaluate_per_prompt_and_store(hyps, refs, prompt_id, language, metrics_dir):
    report = evaluate_generation_metrics(hyps, refs)
    language_dir = os.path.join(metrics_dir, language)
    if not os.path.exists(language_dir):
        os.makedirs(language_dir)
    metrics_file = os.path.join(language_dir, f"prompt_{prompt_id}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    
    print(f"Metrics for prompt {prompt_id} in {language} saved to {metrics_file}")
    return report

if __name__ == "__main__":

    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Total Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        print("CUDA is not available.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="sentence-transformers/LaBSE", type=str)
    parser.add_argument("--gen_model_checkpoint", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--dataset", type=str, default="xlsum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args([])  # Adjusted for notebook execution

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = SentenceTransformer(args.model_checkpoint).to(device)
    gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint)

    output_dir = "generated_responses"
    metrics_dir = os.path.join(output_dir, "metrics")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    selected_languages = ["english", "french", "spanish", "german", "chinese_simplified"]
    sample_sizes = [2, 5, 10, 15, 20]  # Sample sizes to test
    f1_scores = {n: [] for n in sample_sizes}

    start_time = time.time()

    for n in sample_sizes:
        print(f"\nEvaluating for sample size n = {n}")
        dataset = XLSumDataset(sample_size=n)

        for language in tqdm(selected_languages, desc="Processing Selected Languages", unit="language"):
            if language in dataset.train_data:
                messages = construct_prompt(few_shot_examples=[], model_checkpoint=args.gen_model_checkpoint, language=language)
                
                f1_scores_for_n = []
                for prompt_id, message in enumerate(tqdm(messages, desc=f"Processing Prompts in {language}", unit="prompt", leave=False)):
                    response = get_llama3_instruct_chat_response(gen_model, tokenizer, [message], verbose=False)
                    hyps = [response]
                    refs = ["Sample reference text"]
                    
                    prompt_metrics = evaluate_per_prompt_and_store(hyps, refs, prompt_id+1, language, metrics_dir)
                    f1_scores_for_n.append(prompt_metrics["BERTScore (F1)"])

                f1_scores[n].append(np.mean(f1_scores_for_n))

    avg_f1_scores = {n: np.mean(scores) for n, scores in f1_scores.items()}
    plt.plot(list(avg_f1_scores.keys()), list(avg_f1_scores.values()), marker='o')
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Average F1 Score")
    plt.title("F1 Score vs Sample Size")
    plt.show()

    total_time = time.time() - start_time
    print(f"\nAll responses generated and saved successfully in {total_time:.2f} seconds.")
