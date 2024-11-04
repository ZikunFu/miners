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

# Construct prompt for open-ended generation task
def construct_prompt(few_shot_examples, test_tokens, model_checkpoint):
    messages = []
    assistant_role = "assistant"
    if model_checkpoint == "Meta-Llama-3.1-8B-Instruct":
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that performs open-ended generation and outputs completed prompts in multiple languages." 
            "The languages are amharic, arabic, azerbaijani, bengali, burmese, chinese_simplified, chinese_traditional, english, french, gujarati, hausa, hindi, igbo, indonesian, japanese, kirundi, korean, kyrgyz, marathi, nepali, oromo, pashto, persian, pidgin, portuguese, punjabi, russian, scottish_gaelic, serbian_cyrillic, serbian_latin, sinhala, somali, spanish, swahili, tamil, telugu, thai, tigrinya, turkish, ukrainian, urdu, uzbek, vietnamese, welsh, yoruba"
        }
        messages.append(system_message)
    elif model_checkpoint == "google/gemma-2-9b-it":
        assistant_role = "model"
    
    # Add few-shot examples
    for tokens, example_content in few_shot_examples:
        user_message = {
            "role": "user",
            "content": f'''
        "Once upon a time in a world without electricity...",
            "Describe a day in the life of a lighthouse keeper on a remote island.",
            "What do you think the future of transportation will look like in 50 years?",
            "Imagine you are a traveler in ancient China. What sights and experiences would you have?",
            "Write a letter from a time traveler visiting the year 3000 to a friend in the present day.",
            "Describe the most beautiful place you've ever imagined, where nature and technology coexist in harmony.",
            "Tell a story about an artist who discovers a magical paintbrush that brings their creations to life.",
            "Imagine a city where people live in floating homes. Describe a day in the life of one of its residents.",
            "Describe the feelings of the first astronaut to set foot on a newly discovered planet.",
            "Write a tale about a library where each book transports the reader to the story’s world upon opening.",
            "Imagine a world where humans communicate only through music. Describe a conversation.",
            "Create a scene in a world where dreams can be recorded and shared. What does the most popular dream look like?",
            "Describe a festival held by mythical creatures in an enchanted forest.",
            "What would an ordinary school day look like in a school that teaches magic and science side by side?",
            "Imagine you are an explorer in the lost city of Atlantis. What wonders do you encounter?",
            "Write about a village where the trees can talk. What stories do they tell?",
            "Picture a world where every animal can speak. Describe a conversation between two unlikely friends.",
            "Describe a marketplace in a distant galaxy, filled with goods and creatures from across the universe.",
            "Imagine you’ve just met a being who can control time. What advice do they give you?",
            "Tell the story of a friendship between a human and an AI that has gained consciousness.",
            "Write about a world where every person is born with a unique magical ability.",
            "Describe a futuristic city where every building is grown rather than built.",
            "Imagine an annual contest where inventors showcase their most imaginative creations. Describe the winning invention.",
            "Tell a story of a hidden underwater kingdom discovered by deep-sea explorers.",
            "Describe a museum in the future that showcases the extinct technologies of today.",
            "Imagine an artist who paints landscapes that come to life. What’s their most famous work?",
            "Write about a planet where the sun never sets. How do the inhabitants live?",
            "Describe a world where humans live harmoniously with mythical creatures.",
            "Imagine a time-traveling detective solving mysteries throughout history.",
            "Describe a world where every building and vehicle is eco-friendly and alive.",
            "Write about a magical forest where lost things from the human world appear.",
            "Imagine you are a scientist studying alien plant life on a distant planet.",
            "Describe a grand library that contains the knowledge of every civilization in the galaxy.",
            "Tell a story about a music festival that takes place on the moon.",
            "Imagine you’re a translator for the first alien language discovered on Earth. What do they want to tell us?",
            "Write about a machine that can turn dreams into reality. What’s the first wish someone makes?",
            "Describe a futuristic society where robots and humans are close friends.",
            "Imagine a world where people can switch between animal and human form at will.",
            "Tell a tale of a village that celebrates the arrival of each new season with a unique festival.",
            "Describe an enchanted river where the water carries the memories of the past.",
            "Write about an inventor who creates a device to communicate with plants.",
            "Imagine a world where thoughts can be shared directly between minds.",
            "Tell the story of a hidden valley where unicorns live undisturbed by humanity.",
            "Write about a new language that everyone can speak, regardless of their origin.",
            "Imagine a future city where transportation is entirely by flying cars and skybridges.",
            "Describe a magical library where books adapt their story to match the reader’s desires.",
            "Write about a castle in the clouds, only reachable by a hidden staircase.",
            "Imagine you’ve discovered a portal to a parallel world. What’s the first thing you see?",
            "Tell a story about a robot trying to understand human emotions.",
            "Describe a world where people can control the elements, like fire and water.",
            "Imagine a festival where all the inhabitants of Earth’s oceans gather once a year to celebrate."
            '''
        }
        messages.append(user_message)
        assistant_message = {
            "role": assistant_role,
            "content": example_content
        }
        messages.append(assistant_message)
    
    # Add the test sentence
    user_message = {
        "role": "user",
        "content": f'''
       "Once upon a time in a world without electricity...",
        "Describe a day in the life of a lighthouse keeper on a remote island.",
        "What do you think the future of transportation will look like in 50 years?",
        "Imagine you are a traveler in ancient China. What sights and experiences would you have?",
        "Write a letter from a time traveler visiting the year 3000 to a friend in the present day.",
        "Describe the most beautiful place you've ever imagined, where nature and technology coexist in harmony.",
        "Tell a story about an artist who discovers a magical paintbrush that brings their creations to life.",
        "Imagine a city where people live in floating homes. Describe a day in the life of one of its residents.",
        "Describe the feelings of the first astronaut to set foot on a newly discovered planet.",
        "Write a tale about a library where each book transports the reader to the story’s world upon opening.",
        "Imagine a world where humans communicate only through music. Describe a conversation.",
        "Create a scene in a world where dreams can be recorded and shared. What does the most popular dream look like?",
        "Describe a festival held by mythical creatures in an enchanted forest.",
        "What would an ordinary school day look like in a school that teaches magic and science side by side?",
        "Imagine you are an explorer in the lost city of Atlantis. What wonders do you encounter?",
        "Write about a village where the trees can talk. What stories do they tell?",
        "Picture a world where every animal can speak. Describe a conversation between two unlikely friends.",
        "Describe a marketplace in a distant galaxy, filled with goods and creatures from across the universe.",
        "Imagine you’ve just met a being who can control time. What advice do they give you?",
        "Tell the story of a friendship between a human and an AI that has gained consciousness.",
        "Write about a world where every person is born with a unique magical ability.",
        "Describe a futuristic city where every building is grown rather than built.",
        "Imagine an annual contest where inventors showcase their most imaginative creations. Describe the winning invention.",
        "Tell a story of a hidden underwater kingdom discovered by deep-sea explorers.",
        "Describe a museum in the future that showcases the extinct technologies of today.",
        "Imagine an artist who paints landscapes that come to life. What’s their most famous work?",
        "Write about a planet where the sun never sets. How do the inhabitants live?",
        "Describe a world where humans live harmoniously with mythical creatures.",
        "Imagine a time-traveling detective solving mysteries throughout history.",
        "Describe a world where every building and vehicle is eco-friendly and alive.",
        "Write about a magical forest where lost things from the human world appear.",
        "Imagine you are a scientist studying alien plant life on a distant planet.",
        "Describe a grand library that contains the knowledge of every civilization in the galaxy.",
        "Tell a story about a music festival that takes place on the moon.",
        "Imagine you’re a translator for the first alien language discovered on Earth. What do they want to tell us?",
        "Write about a machine that can turn dreams into reality. What’s the first wish someone makes?",
        "Describe a futuristic society where robots and humans are close friends.",
        "Imagine a world where people can switch between animal and human form at will.",
        "Tell a tale of a village that celebrates the arrival of each new season with a unique festival.",
        "Describe an enchanted river where the water carries the memories of the past.",
        "Write about an inventor who creates a device to communicate with plants.",
        "Imagine a world where thoughts can be shared directly between minds.",
        "Tell the story of a hidden valley where unicorns live undisturbed by humanity.",
        "Write about a new language that everyone can speak, regardless of their origin.",
        "Imagine a future city where transportation is entirely by flying cars and skybridges.",
        "Describe a magical library where books adapt their story to match the reader’s desires.",
        "Write about a castle in the clouds, only reachable by a hidden staircase.",
        "Imagine you’ve discovered a portal to a parallel world. What’s the first thing you see?",
        "Tell a story about a robot trying to understand human emotions.",
        "Describe a world where people can control the elements, like fire and water.",
        "Imagine a festival where all the inhabitants of Earth’s oceans gather once a year to celebrate."
        '''
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
