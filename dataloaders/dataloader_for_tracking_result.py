from collections import namedtuple
# System imports
import os
import random

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, Sampler
import torch_geometric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
def load_graph(filename):

    with np.load(filename) as f:
        trigger_flag = f['trigger']
        edge_index = f['edge_index']
        hits = f['hits']
        e = f['e']

    return hits, edge_index, trigger_flag, e

class HitGraphDataset():
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None, real_weight=1.0, false_weight=1.0,
                n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True, n_samples2=None,
                n_samples3=None):
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist))
            filenames = self.metadata.file.values
        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            self.filenames = filenames if n_samples is None else filenames[:n_samples]
            if n_input_dir == 2:
                input_dir2 = os.path.expandvars(input_dir2)
                filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
                self.filenames += (filenames if n_samples2 is None else filenames[:n_samples2])
            if n_input_dir == 3:
                input_dir2 = os.path.expandvars(input_dir2)
                filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
                self.filenames += (filenames if n_samples2 is None else filenames[:(n_samples2)])

                input_dir3 = os.path.expandvars(input_dir3)
                filenames = sorted([os.path.join(input_dir3, f) for f in os.listdir(input_dir3)
                                if f.startswith('event')])
                self.filenames += (filenames if n_samples3 is None else filenames[:(n_samples3)])
            random.shuffle(self.filenames)
        else:
            raise Exception('Must provide either input_dir or filelist to HitGraphDataset')
        
        self.random_permutation = random_permutation
        self.hits_info = []
        self.edge_index = []
        self.trig = []
        self.e = []
        self.n_hits = []

        for file_index in range(len(self.filenames)):
            hits, edge_index, trigger_flag, e = load_graph(self.filenames[file_index])
            self.hits_info.append(hits)
            self.edge_index.append(edge_index)
            self.trig.append(trigger_flag)
            self.e.append(e)
            self.n_hits.append(hits.shape[0])

        self.n_hits = np.array(self.n_hits)
    

    def __getitem__(self, index):
        return torch_geometric.data.Data(x=torch.from_numpy(self.hits_info[index]), \
            edge_index=torch.from_numpy(self.edge_index[index].astype(np.long)),\
            trigger=torch.from_numpy(self.trig[index]), \
            n_hits=torch.from_numpy(np.array(self.n_hits[index])), \
            e=torch.from_numpy(np.array(self.e[index])))

    def __len__(self):
        return len(self.filenames)

def get_data_loaders(name, batch_size, model_name, **data_args):
    if name == 'hits_loader':
        train_dataset, valid_dataset, train_data_n_nodes, valid_data_n_nodes = get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
    
    from torch_geometric.data import Batch
    collate_fn = Batch.from_data_list
    if model_name == 'GNNPairDiffpool':
        train_batch_sampler = JetsBatchSampler(train_data_n_tracks, batch_size)
        train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
        valid_batch_sampler = JetsBatchSampler(valid_data_n_tracks, batch_size)
        valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
    
    else:
        train_batch_sampler = JetsBatchSampler(train_data_n_nodes, batch_size)
        train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
        valid_batch_sampler = JetsBatchSampler(valid_data_n_nodes, batch_size)
        valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=collate_fn)
    return train_data_loader, valid_data_loader


def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, false_weight=1.0,
                n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True,
                n_train2=None, n_valid2=None, n_train3=None, n_valid3=None):
    data = HitGraphDataset(input_dir=input_dir, filelist=filelist,
                           n_samples=n_train+n_valid, real_weight=real_weight,
                           false_weight=false_weight,
                           n_input_dir=n_input_dir, input_dir2=input_dir2, 
                           input_dir3=input_dir3, random_permutation=random_permutation,
                           n_samples2=n_train2+n_valid2, n_samples3=n_train3+n_valid3)
    # Split into train and validation
    if n_input_dir == 2:
        train_data, valid_data = random_split(data, [n_train+n_train2, n_valid+n_valid2])
    elif n_input_dir == 3:
        train_data, valid_data = random_split(data, [n_train+n_train2+n_train3, n_valid+n_valid2+n_valid3])
    else:
        train_data, valid_data = random_split(data, [n_train, n_valid])

    train_data_n_nodes = data.n_hits[train_data.indices]
    valid_data_n_nodes = data.n_hits[valid_data.indices]
    return train_data, valid_data, train_data_n_nodes, valid_data_n_nodes

class JetsBatchSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the jets
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i <= 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = np.ceil(max(n_of_size / self.batch_size, 1))

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]
