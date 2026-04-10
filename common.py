from contextlib import contextmanager
from time import perf_counter

import torch
from torch.utils.data import DataLoader

class perf_timer:
    '''Context manager for timing a block of code.'''
    
    def __init__(self):
        self.paused = 0

    def __enter__(self):
        self.start = perf_counter()
        return self
    
    def __exit__(self, *exc):
        self.end = perf_counter()
    
    @property
    def unpaused(self):
        return self.end - self.start

    @property
    def total(self):
        return self.unpaused - self.paused
    
    @contextmanager
    def pause(self):
        start = perf_counter()
        yield
        end = perf_counter()
        self.paused += end - start

def sparse_collate(batch):
    """Custom collate for sparse CSR tensors"""
    sparse_Xs, ys = zip(*batch)
    print(sparse_Xs)
    return torch.tensor(sparse_Xs), torch.tensor(ys)

def data_loader(*argv, **kwargs):
    return DataLoader(*argv, **kwargs, collate_fn=sparse_collate)