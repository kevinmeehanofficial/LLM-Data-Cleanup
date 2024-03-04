"""
Removes every other sequence between two end tokens in a file to make data lighter or remove similar data in noisy, repetitive datasets
"""

import numpy as np
import mmap
from tqdm import tqdm

def remove_every_other_sequence(input_file_path, end_token):
    TOKEN_TYPE = np.uint16
    TOKEN_SIZE = np.dtype(TOKEN_TYPE).itemsize

    with open(input_file_path, 'r+b') as f_in:
        mm_in = mmap.mmap(f_in.fileno(), 0)
        total_tokens = mm_in.size() // TOKEN_SIZE
        chunk = np.ndarray(buffer=mm_in, dtype=TOKEN_TYPE, shape=(total_tokens,))
        print(f"Total tokens in the origional file: {len(chunk)}")

        end_indices = np.where(chunk == end_token)[0]
        indices_to_keep = np.ones(len(chunk), dtype=bool)

        # Iterate over sequences and remove every other one
        remove_flag = True  # Start with removal for the first sequence
        last_index = 0
        for end_index in tqdm(end_indices, desc="Processing Sequences"):
            if remove_flag:
                indices_to_keep[last_index:end_index + 1] = False
            last_index = end_index + 1
            remove_flag = not remove_flag

        # Create final chunk and update the file
        final_chunk = chunk[indices_to_keep]
        new_size = len(final_chunk) * TOKEN_SIZE
        mm_in[:new_size] = final_chunk.tobytes()
        mm_in.flush()
        mm_in.close()
        f_in.truncate(new_size)

        print("Processing complete.")
        print(f"Total tokens in the new file: {len(final_chunk)}")
        return final_chunk


# Example usage
end_token = 126    # Replace with actual end token    
input_file_path = "/your/path/here"  # Replace with actual file path
processed_chunk = remove_every_other_sequence(input_file_path, end_token)
