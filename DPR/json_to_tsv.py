import json
import csv
import sys

def json_to_tsv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as json_file, open(output_file, 'w', encoding='utf-8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')

        for line in json_file:
            # Parse each JSON line
            data = json.loads(line)
            
            # Extract fields for the TSV format
            tsv_row = f"{data['id']}#0", data['question']
            tsv_writer.writerow(tsv_row)

    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Check if file paths are provided
    if len(sys.argv) < 2:
        print("Usage: python json_to_tsv.py <input_json_file1> <input_json_file2> ...")
    else:
        # Loop through each provided file
        for input_file in sys.argv[1:]:
            # Define output file name based on input file name
            output_file = input_file.replace('.json', '.tsv')
            json_to_tsv(input_file, output_file)

