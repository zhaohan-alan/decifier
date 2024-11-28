import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class DeciferDataset(Dataset):
    def __init__(self, h5_path, data_keys):
        self.h5_file = h5py.File(h5_path, 'r')
        self.data_keys = data_keys

        # Ensure that data_keys only contain datasets
        self.data = {}
        for key in self.data_keys:
            item = self.h5_file[key]
            if isinstance(item, h5py.Dataset):
                self.data[key] = item
            else:
                raise TypeError(f"The key '{key}' does not correspnd to an h5py.Dataset.")

        self.dataset_length = len(next(iter(self.data.values())))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        data = {}
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

            data[key] = sequence

        return data
