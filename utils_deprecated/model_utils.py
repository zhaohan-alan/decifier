import torch
from torch.utils.data import BatchSampler
import random

class RandomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter(self):
        # Each time __iter__ is called, radomize the batch indices
        batch_indices = list(self.sampler)
        random.shuffle(batch_indices)

        # Return batches of size batch_Size
        for i in range(0, len(batch_indices), self.batch_size):
            yield batch_indices[i:i + self.batch_size]
