"""
When working on a classification task with an LLM, you may need to label data with tokens that represent the classification of the tokens before it.

For example:

Initial Tokens: 10 (start of sequence), 20, 30, 24, 34, 96 (end of input sequence)
Classification: 119 (or 120), 22352

After adding your classifications to the initial tokens, you can use this script to confirm that the token you inserted correctly shows up after the token you inserted it behind.

This script counts all the occurrences of the tokens listed in "TOKEN_MAPPING" after a specific token.

"""

import numpy as np
import mmap

def check_token_sequence(input_file_path, target_tokens, preceding_tokens):
    with open(input_file_path, 'rb') as f_in:
        mm_in = mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ)
        TOKEN_TYPE = np.uint16
        TOKEN_SIZE = np.dtype(TOKEN_TYPE).itemsize
        total_tokens = mm_in.size() // TOKEN_SIZE
        chunk = np.ndarray(buffer=mm_in, dtype=TOKEN_TYPE, shape=(total_tokens,))

        occurrences = {}
        # Find indices of preceding tokens
        preceding_indices = np.where(np.isin(chunk[:-1], preceding_tokens))[0]
        
        for token_name, target_token in target_tokens.items():
            # Check if next token is each target token
            target_follows = chunk[preceding_indices + 1] == target_token
            sequence_count = np.sum(target_follows)
            occurrences[token_name] = sequence_count
        
        mm_in.close()
        return occurrences

TOKEN_MAPPING = {
    '12': 22352,
    '13': 22433,
    '14': 22361,
    '15': 22425,
    '16': 22424,
    '17': 22402,
    '18': 22441,
    '19': 22360,
    '20': 22347,
    '21': 22335,
    '22': 22342,
    '23': 22368,
    '24': 22353,
    '25': 22409,
    '26': 22392,
    '27': 22374,
    '28': 22351,
    '29': 22417,
    '30': 22345,
    '31': 22418,
    '32': 22344,
    '33': 22414,
    '34': 22411,
    '35': 22439,
    '36': 22430,
    '37': 22435,
    '38': 22348,
}

input_file_path = "/home/kevin/Projects/llama2.c/data/AlphaStratAI/backupdata/20000_to_45000_restructured_for_round2/train_split_1_relabeled.bin"
preceding_tokens = [119, 120] # Check for the occurrence of each of the tokens in token mapping, which occurs after any of these tokens.

# Find and print occurrences for each token
occurrences = check_token_sequence(input_file_path, TOKEN_MAPPING, preceding_tokens)

for token_name, count in occurrences.items():
    print(f"Occurrences of token {TOKEN_MAPPING[token_name]} (ID: {token_name}) directly after 119 or 120: {count}")
    
