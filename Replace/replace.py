"""
Replaces all occurrences of old_token with new_token in a .bin file using memory mapping,
and prints the number of occurrences of the token before and after the replacement.
    
Parameters:
file_path (str): Path to the .bin file.
dtype (numpy.dtype): Data type of the array stored in the file.
old_token (uint16): The token value to be replaced.
new_token (uint16): The token value to replace with.
"""


import numpy as np
import os
from tqdm import tqdm

def replace_token_in_mapped_file(file_path, dtype, old_token, new_token):

    # Ensure tokens are of uint16 type
    old_token = np.uint16(old_token)
    new_token = np.uint16(new_token)
    
    # Determine the number of elements in the file based on its size
    file_size = os.path.getsize(file_path)
    num_elements = file_size // dtype().itemsize

    # Open the file in read+write mode using memory mapping
    mmap = np.memmap(file_path, dtype=dtype, mode='r+', shape=(num_elements,))

    # Count occurrences of the old token before replacement
    old_token_count_before = np.count_nonzero(mmap == old_token)
    new_token_count_before = np.count_nonzero(mmap == new_token)    
    print(f"Occurrences of the old token {old_token} before replacement: {old_token_count_before}")
    print(f"Occurrences of the new token {new_token} before replacement: {new_token_count_before}")

    # Use tqdm for a progress bar during replacement
    with tqdm(total=num_elements, desc="Processing") as pbar:
        for i in range(num_elements):
            if mmap[i] == old_token:
                mmap[i] = new_token
            pbar.update()

    # Count occurrences of the old token after replacement
    old_token_count_after = np.count_nonzero(mmap == old_token)
    new_token_count_after = np.count_nonzero(mmap == new_token)   
    print(f"Occurrences of the old token {old_token} after replacement: {old_token_count_after}")    
    print(f"Occurrences of the new token {new_token} after replacement: {new_token_count_after}")

    # Sync changes to disk and close the memmap
    mmap.flush()
    del mmap

# Example usage
if __name__ == "__main__":
    # Define file path and parameters
    file_path = 'your/file/path/here'  # Update this to your .bin file path
    dtype = np.uint16  # Data type of the array
    
    # Replace all occurrences of 22560 with 22559
    replace_token_in_mapped_file(file_path, dtype, 22560, 22559)
