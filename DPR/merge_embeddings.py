import argparse
import glob
import os
import faiss
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, help='dimension of passage embeddings', default=768)
parser.add_argument('--input', type=str, help='wildcard directory pattern for input indexes', required=True)
parser.add_argument('--output', type=str, help='directory to output the full index', required=True)
args = parser.parse_args()

# Create the output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Create a new Faiss index for merging
new_index = faiss.IndexFlatIP(args.dimension)
docid_files = []

# Merge all shard indexes
for index_dir in tqdm(sorted(glob.glob(args.input)), desc="Merging Faiss Index"):
    index = faiss.read_index(os.path.join(index_dir, 'index'))
    docid_files.append(os.path.join(index_dir, 'docid'))
    vectors = index.reconstruct_n(0, index.ntotal)
    new_index.add(vectors)

# Write the merged index
faiss.write_index(new_index, os.path.join(args.output, 'index'))

# Merge docid files into a single file
with open(os.path.join(args.output, 'docid'), 'w') as wfd:
    for f in docid_files:
        with open(f, 'r') as f1:
            for line in f1:
                wfd.write(line)

