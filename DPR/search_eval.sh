#!/bin/bash
#SBATCH --output=recall_%j.log           # Standard output and error log (%j creates a unique ID)
#SBATCH --time=24:00:00                  # Set a maximum time limit (24 hours)
#SBATCH --gpus=1                         # Number of GPUs required
#SBATCH --mem-per-gpu=12G                # Memory per GPU
#SBATCH --cpus-per-task=2                # Number of CPU cores per task (adjust as needed)
#SBATCH --open-mode=append

corpus_name=$1
enc_model=$2
lang=$3

source /home/mgoyani/env/bin/activate
cd /home/mgoyani/scratch
module load StdEnv/2023 gcc cuda faiss/1.7.4
module load java/21.0.1

python -m pyserini.search.faiss \
  --threads 16 --batch-size 16 \
  --encoder-class auto \
  --encoder ${enc_model} \
  --topics miracl-v1.0-${lang}-dev \
  --index /home/mgoyani/scratch/indexes/${corpus_name}_${enc_model} \
  --output /home/mgoyani/scratch/output/${corpus_name}_${enc_model}_${lang}.txt --hits 1000

python -m pyserini.eval.trec_eval \
  -c -m recall.100 miracl-v1.0-${lang}-dev \
  /home/mgoyani/scratch/output/${corpus_name}_${enc_model}_${lang}.txt

