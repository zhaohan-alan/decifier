import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from . import Tokenizer
import sys
sys.path.append("./")

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, data_to_load, block_size, numeric_padding_value=0):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.data_to_load = data_to_load
        self.block_size = block_size
        self.data = {key: self.h5_file[key] for key in self.data_to_load}
        
        self.token_padding_value = Tokenizer().padding_id
        self.numeric_padding_value = numeric_padding_value

    def __len__(self):
        total_chunks = 0
        for key in self.data_to_load:
            for sequence in self.data[key]:
                total_chunks += (len(sequence) + self.block_size - 1) // self.block_size
        return total_chunks

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
        total_chunks = 0
        for key in self.data_to_load:
            for i, sequence in enumerate(self.data[key]):
                num_chunks = (len(sequence) + self.block_size - 1) // self.block_size
                if idx < total_chunks + num_chunks:
                    chunk_idx = idx - total_chunks
                    return {'sequence_idx': i, 'chunk_idx': chunk_idx}
                total_chunks += num_chunks
        raise IndexError("Index out of bounds")