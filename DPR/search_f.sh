#!/bin/bash
#SBATCH --output=recall_%j.log           # Standard output and error log (%j creates a unique ID)
#SBATCH --time=24:00:00                  # Set a maximum time limit (24 hours)
#SBATCH --gpus=1                         # Number of GPUs required
#SBATCH --mem-per-gpu=12G                # Memory per GPU
#SBATCH --cpus-per-task=2                # Number of CPU cores per task (adjust as needed)
#SBATCH --open-mode=append


source /home/mgoyani/env/bin/activate
cd /home/mgoyani/scratch
module load StdEnv/2023 gcc cuda faiss/1.7.4
module load java/21.0.1

python -m pyserini.search.faiss \
  --threads 16 --batch-size 16 \
  --encoder-class auto \
  --encoder sentence-transformers/LaBSE \
  --topics /home/mgoyani/scratch/afriqa/data/queries/fon/queries.afriqa.fon.fr.dev.tsv \
  --index /home/mgoyani/scratch/indexes/fr_LaBSE \
  --output /home/mgoyani/scratch/output/fr_LaBSE_fon.txt --hits 1000

python -m pyserini.eval.trec_eval \
  -c -m recall.100 /home/mgoyani/scratch/afriqa/data/queries/fon/queries.afriqa.fon.fr.dev.tsv \
  /home/mgoyani/scratch/output/fr_LaBSE_fon.txt
