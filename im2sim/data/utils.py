import torch
from torch_geometric.data import Data, Batch
from functools import partial
import numpy as np

def collate(batch):
    out_dict = {}
    for key,val in batch[0].items():
        if isinstance(val, torch.Tensor):
            out_dict[key] = torch.stack([b[key] for b in batch])
        elif isinstance(val, Data):
            out_dict[key] = Batch.from_data_list([b[key] for b in batch])
        else:
            raise TypeError(f"{key} is type {type(val)}. Generator outputs must be either torch.Tensor or torch_geometric.data.Data object")
    return out_dict

DataLoader = partial(torch.utils.data.DataLoader, collate_fn = collate)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, load_fn, cases, transform=None):
        self.load_fn = load_fn
        self.cases = cases
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        sample = self.load_fn(case)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
  
