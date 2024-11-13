import json
import os

# Configuration
input_file = "/home/mgoyani/scratch/corpus/french.jsonl"  # Path to the original dataset
output_dir = "/home/mgoyani/scratch/corpus/french"               # Directory to save the chunks
num_chunks = 10                                           # Number of chunks

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Calculate the number of lines per chunk
with open(input_file, 'r') as f:
    total_lines = sum(1 for _ in f)
lines_per_chunk = total_lines // num_chunks

# Split and save the chunks
with open(input_file, 'r') as f:
    for chunk_id in range(num_chunks):
        output_file = os.path.join(output_dir, f"corpus_chunk_{chunk_id}.jsonl")
        with open(output_file, 'w') as out_f:
            for i, line in enumerate(f):
                if i >= lines_per_chunk and chunk_id < num_chunks - 1:
                    break
                out_f.write(line)
        print(f"Chunk {chunk_id + 1}/{num_chunks} saved to {output_file}")

print("Dataset chunking completed.")

