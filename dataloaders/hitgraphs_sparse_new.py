"""Dataset specification for hit graphs using pytorch_geometric formulation"""

# System imports
import os

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric

def load_graph(filename):
    with np.load(filename) as f:
        x = f['scaled_hits'][:, :3]
        edge_index = f['edge_index']
        start, end = edge_index
        pid = f['pid']
        y = np.logical_and(pid[start] != -1, pid[start] == pid[end])
        trigger = f['trigger']
    return x, edge_index, y, trigger

class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None, real_weight=1.0, n_folders=1, input_dir2=None):
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist))
            filenames = self.metadata.file.values
        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
        else:
            raise Exception('Must provide either input_dir or filelist to HitGraphDataset')
        self.filenames = filenames if n_samples is None else filenames[:n_samples]
        if n_folders == 2:
            filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            self.filenames += filenames[:n_samples]
        self.real_weight = real_weight
        self.fake_weight = 1 #real_weight / (2 * real_weight - 1)

    def __getitem__(self, index):
        x, edge_index, y, trigger = load_graph(self.filenames[index])
        # print('before x:', x)
        # n_MVTX = np.sum(x[:, 3] != -1)
        # x = x[x[:, 3] == -1, :3]
        # print('after x:', x)
        # print(edge_index)
        # edge_index = edge_index - n_MVTX
        # mask = (edge_index[0, :] >= 0)
        # edge_index = edge_index[:, mask]
        # y = y[mask]
        # print(edge_index)
        # print(n_MVTX)
        # print('#########################')
        # Compute weights
        w = y * self.real_weight + (1-y) * self.fake_weight
        return torch_geometric.data.Data(x=torch.from_numpy(x),
                                         edge_index=torch.from_numpy(edge_index),
                                         y=torch.from_numpy(y), w=torch.from_numpy(w),
                                         i=index, trigger=torch.from_numpy(trigger))

    def __len__(self):
        return len(self.filenames)

def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, n_folders=1, input_dir2=None):
    data = HitGraphDataset(input_dir=input_dir, filelist=filelist,
                           n_samples=n_train+n_valid, real_weight=real_weight, n_folders=n_folders, input_dir2=input_dir2)
    # Split into train and validation
    if n_folders == 1:
        train_data, valid_data = random_split(data, [n_train, n_valid])
    if n_folders == 2:
        train_data, valid_data = random_split(data, [2*n_train, 2*n_valid])
    return train_data, valid_data
