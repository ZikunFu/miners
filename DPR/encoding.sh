#!/bin/bash
#SBATCH --output=output_%j.log           # Standard output and error log (%j creates a unique ID)
#SBATCH --time=24:00:00                  # Set a maximum time limit (24 hours)
#SBATCH --gpus=1                         # Number of GPUs required
#SBATCH --mem-per-gpu=12G                # Memory per GPU
#SBATCH --cpus-per-task=2                # Number of CPU cores per task (adjust as needed)
#SBATCH --open-mode=append

corpus_name=$1
enc_model=$2

source /home/mgoyani/env/bin/activate
cd /home/mgoyani/scratch
module load StdEnv/2023 gcc cuda faiss/1.7.4
module load java/21.0.1

python -m pyserini.encode \
    input   --corpus /home/mgoyani/scratch/corpus/${corpus_name}.jsonl \
            --fields title text \
            --delimiter "\n\n" \
            --shard-id 0 \
            --shard-num 1 \
    output  --embeddings /home/mgoyani/scratch/indexes/${corpus_name}_${enc_model} \
            --to-faiss \
    encoder --encoder ${enc_model} \
            --fields title text \
            --batch 32 \
            --encoder-class 'auto' \
            --fp16

