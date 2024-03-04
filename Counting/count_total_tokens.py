"""
A simple script that counts all of the tokens in a .bin file.

Replace the dtype with the approprate dtype for your file.

"""

import numpy as np

def count_uint16_tokens(file_path):
    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the entire file content into a NumPy array with dtype=np.uint16
        data = np.fromfile(file, dtype=np.uint16)
        
    # Print the count of uint16 tokens
    print(f"Count of dtype=np.uint16 tokens in the file: {data.size}")

# Example usage
file_path = '/your/path/here'  # Replace 'your_file_path.bin' with your actual file path
count_uint16_tokens(file_path)



