"""
shuffles.py 

Shuffling data between training runs is standard practice when training large language models.

This script takes a starting token (used to indicate the start of each prompt/ sequence of tokens) and uses this token to determine each sequence of tokens.

It then randomly shuffles the sequences of tokens and dumps them into a new .bin file. 

"""

import numpy as np
import random
from tqdm import tqdm

def extract_sequences(data):
    sequences = []
    start_indices = np.where(data == start_token)[0]
    end_indices = np.append(start_indices[1:], len(data))
    
    for start, end in tqdm(zip(start_indices, end_indices), total=len(start_indices), desc="Extracting sequences"):
        sequence = data[start:end]  # End is exclusive, so it stops before the next token 10
        sequences.append(sequence)
    
    return sequences

def write_random_sequences(sequences, output_file_path):
    random.shuffle(sequences)  # Shuffle the sequences randomly
    with open(output_file_path, 'wb') as f:
        for sequence in tqdm(sequences, desc="Writing sequences"):
            f.write(sequence.tobytes())  # Write each sequence to the file

def process_file(input_file_path, output_file_path):
    # Read the input file
    with open(input_file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)  # Assuming the data type is uint16; adjust as necessary
    
    # Extract sequences
    sequences = extract_sequences(data)
    
    # Write the shuffled sequences to the output file
    write_random_sequences(sequences, output_file_path)

# Example usage with a real .bin file
start_token = 10
input_file_path = 'path/to/your/input_file.bin'
output_file_path = 'path/to/your/output_file.bin'
process_file(input_file_path, output_file_path)


