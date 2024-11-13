#!/bin/bash
#SBATCH --job-name=pyserini         # Job name
#SBATCH --output=pyserini_%j.log    # Standard output and error log (%j creates a unique ID)
#SBATCH --time=48:00:00                           # Set a maximum time limit (48 hours)
#SBATCH --gpus=1                                  # Number of GPUs required
#SBATCH --mem-per-gpu=12G                         # Memory per GPU
#SBATCH --open-mode=append

source /home/mgoyani/miners/bin/activate

# Run the Python script
python /home/mgoyani/masakhane_xqa/baselines/retriever/dense/pyserini/search.py \
        --topics '/home/mgoyani/masakhane_xqa/data/queries/bem/queries.afriqa.bem.en.train.tsv' \
        --index  'wikipedia-dpr-100w.dpr-multi' \
        --encoder 'sentence-transformers/LaBSE' \
        --encoder-class auto \
        --batch-size 128 \
        --threads 12 \
        --output '/home/mgoyani/masakhane_xqa/queries/bem_train.trec'
