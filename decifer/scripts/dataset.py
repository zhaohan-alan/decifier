import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from . import Tokenizer
import sys
sys.path.append("./")
from time import time
from tqdm.auto import tqdm

class DeciferDataset(Dataset):
    def __init__(self, h5_path, data_keys):
        self.h5_file = h5py.File(h5_path, 'r')
        self.data_keys = data_keys
        self.data = {key: self.h5_file[key] for key in self.data_keys}
        self.dataset_length = len(next(iter(self.data.values())))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        data = []
        for key in self.data_keys:
            sequence = self.data[key][idx]

            # Handle numeric data (np.ndarray)
            if isinstance(sequence, np.ndarray):
                dtype = torch.float32 if 'float' in str(sequence.dtype) else torch.long
                sequence = torch.tensor(sequence, dtype=dtype)
            elif isinstance(sequence, (bytes, str)):
                sequence = sequence.decode('utf-8') if isinstance(sequence, bytes) else sequence
            else:
                raise TypeError(f"Unsupported sequence type {type(sequence)}")

            data.append(sequence)

        return tuple(data)

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, data_to_load, block_size, numeric_padding_value=0):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.data_to_load = data_to_load
        self.block_size = block_size
        self.data = {key: self.h5_file[key] for key in self.data_to_load}
        
        self.token_padding_value = Tokenizer().padding_id
        self.numeric_padding_value = numeric_padding_value
        
        # Precompute chunk offsets with a progress bar
        self.sequence_chunk_map = []
        total_chunks = 0
        
        # Use tqdm to wrap the outer loop and provide a progress bar
        for key in tqdm(self.data_to_load, desc="Processing datasets", leave=False):
            for i, sequence in enumerate(self.data[key]):
                num_chunks = (len(sequence) + block_size - 1) // block_size
                self.sequence_chunk_map.append((key, i, total_chunks, num_chunks))
                total_chunks += num_chunks

        self.total_chunks = total_chunks

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        sequence_chunks = self.find_sequence_and_chunk(idx)
        data = []
        block_size = self.block_size  # Cache the block size for faster access

        for key in self.data_to_load:
            sequence = self.data[key][sequence_chunks['sequence_idx']]
            chunk = None

            # Handle tokenized data
            if 'tokenized' in key:
                start = sequence_chunks['chunk_idx'] * block_size
                end = min(start + block_size, len(sequence))
                chunk = sequence[start:end]
            else:
                chunk = sequence

            # Handle numeric data (np.ndarray)
            if isinstance(chunk, np.ndarray):
                dtype = torch.float32 if 'float' in str(chunk.dtype) else torch.long
                chunk = torch.tensor(chunk, dtype=dtype)

            # Handle tokenized data (strings)
            elif isinstance(chunk, (bytes, str)):
                chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk

            # Pad if necessary
            if isinstance(chunk, torch.Tensor) and len(chunk) < block_size:
                pad_value = self.token_padding_value if chunk.dtype == torch.long else self.numeric_padding_value
                chunk = torch.cat([chunk, torch.full((block_size - len(chunk),), pad_value, dtype=chunk.dtype)], dim=0)

            data.append(chunk)

        return tuple(data)

    def find_sequence_and_chunk(self, idx):
        # Binary search to find the correct sequence and chunk
        left, right = 0, len(self.sequence_chunk_map) - 1
        while left <= right:
            mid = (left + right) // 2
            key, seq_idx, chunk_start, num_chunks = self.sequence_chunk_map[mid]
            
            if chunk_start <= idx < chunk_start + num_chunks:
                chunk_idx = idx - chunk_start
                return {'sequence_idx': seq_idx, 'chunk_idx': chunk_idx}
            elif idx < chunk_start:
                right = mid - 1
            else:
                left = mid + 1
        raise IndexError("Index out of bounds")
