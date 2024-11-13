#!/bin/bash
#SBATCH --output=merge_output.log
#SBATCH --time=1:00:00
#SBATCH --mem=32G  # Memory needed for merging

source /home/mgoyani/env/bin/activate
cd /home/mgoyani/scratch
module load StdEnv/2023 gcc cuda faiss/1.7.4
module load java/21.0.1

# Run the merging script
python merge_embeddings.py \
    --input "/home/mgoyani/scratch/ind_shards/*" \
    --output "/home/mgoyani/scratch/ind" \
    --dimension 768

