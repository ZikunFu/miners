#!/bin/bash
#SBATCH --job-name=encode_corpus
#SBATCH --output=encoding_%A_%a.log          # Unique log file for each array job
#SBATCH --time=24:00:00                      # Maximum time limit (24 hours)
#SBATCH --gpus=1                             # Number of GPUs per job
#SBATCH --mem=16G                            # Memory per GPU
#SBATCH --cpus-per-task=5                    # Number of CPU cores per task
#SBATCH --array=0-9                          # Array of jobs, one per shard

# Paths
CORPUS_DIR="/home/mgoyani/scratch/corpus/french"  # Directory containing chunked corpus
OUTPUT_DIR="/home/mgoyani/scratch/ind_shards"     # Directory to store shard indexes

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

source /home/mgoyani/env/bin/activate
cd /home/mgoyani/scratch
module load StdEnv/2023 gcc cuda faiss/1.7.4
module load java/21.0.1

# Get the file name based on the job array ID
INPUT_FILE="${CORPUS_DIR}/corpus_chunk_${SLURM_ARRAY_TASK_ID}.jsonl"
OUTPUT_SHARD="${OUTPUT_DIR}/shard_${SLURM_ARRAY_TASK_ID}"

# Run the encoding process
python -m pyserini.encode \
    input --corpus "$INPUT_FILE" \
          --fields text title \
          --delimiter "\n\n" \
          --shard-id $SLURM_ARRAY_TASK_ID \
          --shard-num 10 \
    output --embeddings "$OUTPUT_SHARD" \
           --to-faiss \
    encoder --encoder sentence-transformers/LaBSE \
            --fields text title \
            --batch 2 \
            --encoder-class 'auto' \
            --fp16

