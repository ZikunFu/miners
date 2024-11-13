import json
from tqdm import tqdm

# Input and output file paths
input_file_path = '/home/mgoyani/scratch/corpus/french.jsonl'     #len = 28,360,592
output_file_path = '/home/mgoyani/scratch/corpus/fr.jsonl'


if __name__ == "__main__":
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in tqdm(infile):
            doc = json.loads(line)  # Load each line as a JSON object

            # Transform to the expected format
            jsonl_doc = {
                "id": str(doc['docid']),  # Convert 'docid' to string for 'id' field
                "contents": f"{doc['text']}\n\n{doc['title']}"  # Combine 'title' and 'text' with '\n'
            }

            # Write the transformed document to the output file
            json.dump(jsonl_doc, outfile)
            outfile.write('\n')
