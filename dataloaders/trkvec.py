"""Dataset specification for track sets using pytorch_geometric formulation"""

# System imports
import os
import math

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric
from sklearn.neighbors import NearestNeighbors

def load_graph(filename, load_compelete_graph=False):

    with np.load(filename) as f:
        complete_flags = f['complete_flags'] 
        if load_compelete_graph and len(complete_flags)!=0:
            track_vector = f['track_vector'][complete_flags]
            origin_vertices = f['origin_vertices'][complete_flags]
            momentums = f['momentums'][complete_flags]
            pids = f['pids'][complete_flags]
            ptypes = f ['ptypes'][complete_flags]
            energy = f['energy'][complete_flags]
        else:
            track_vector = f['track_vector']
            origin_vertices = f['origin_vertices']
            momentums = f['momentums']
            pids = f['pids']
            ptypes = f ['ptypes']
            energy = f['energy']
        trigger = f['trigger']
        ip = f['ip']
        n_track = track_vector.shape[0]
        if n_track != 0:
            adj = (origin_vertices[:,None] == origin_vertices).all(axis=2)
        else:
            adj = 0
    return track_vector, complete_flags, origin_vertices, momentums, pids, ptypes, energy, trigger, ip, adj

def get_distance(x, y):
    x1, x2 = x[0:3], x[3:6]
    y1, y2 = y[0:3], y[3:6]
    return np.abs(distance_from_two_lines(x2-x1, y2-y1, x1, y1))

def distance_from_two_lines(e1, e2, r1, r2):
    n = np.cross(e1, e2)
    if np.linalg.norm(n) == 0:
        return np.linalg.norm(np.cross(r2 - r1, e1)) / np.linalg.norm(e2)
    n /= np.linalg.norm(n)
    # Calculate distance
    d = np.dot(n, r1 - r2)
    return d

def get_dca(X):
    n = X.shape[0]
    dca = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dca[i, j] = get_distance(X[i], X[j])
    return dca


def sigmoid(x):
     return 1 / (1 + math.exp(-x))

class TrkDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None, real_weight=1.0, n_folders=1, input_dir2=None, knn_k=20, adj_mode=None, load_complete_graph=False, threshold=np.exp(-1.5)):
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
        self.knn_k = knn_k
        self.adj_mode = adj_mode
        self.load_complete_graph = load_complete_graph
        self.threshold = threshold

    def __getitem__(self, index):
        track_vector, complete_flags, origin_vertices, momentums, pids, ptypes, energy, trigger, ip, adj = load_graph(self.filenames[index], self.load_complete_graph)
        if self.adj_mode == 'knn':
            nbrs = NearestNeighbors(n_neighbors=min(track_vector.shape[0], self.knn_k), metric=get_distance)
            nbrs.fit(track_vector)
            _, knn_adj = nbrs.kneighbors(track_vector)
            term = np.repeat(np.arange(knn_adj.shape[0]), knn_adj.shape[1], axis=0)
            knn_adj = knn_adj.reshape(-1)
            edge_index = np.array(list(zip(term, knn_adj))).transpose()
            return torch_geometric.data.Data(x=torch.from_numpy(track_vector).float(),
                                            edge_index=torch.from_numpy(edge_index),
                                            i=index, trigger=torch.from_numpy(trigger))
        if self.adj_mode == 'threshold':
            dca = get_dca(track_vector)
            dca[dca > self.threshold] = 0
            dca = torch.from_numpy(dca)
            index = dca.nonzero(as_tuple=True)
            edge_attr = (self.threshold - dca[index])/self.threshold # mark 0 - threshold to 0 - 1
            edge_index = torch.stack(index, dim=0)
            return torch_geometric.data.Data(x=torch.from_numpy(track_vector).float(),
                                         edge_index=edge_index,
                                         edge_attr=edge_attr,
                                         i=index, trigger=torch.from_numpy(trigger))

    def __len__(self):
        return len(self.filenames)

def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, n_folders=1, input_dir2=None, knn_k=20, adj_mode=None, load_complete_graph=False):
    data = TrkDataset(input_dir=input_dir, filelist=filelist,
                           n_samples=n_train+n_valid, real_weight=real_weight, n_folders=n_folders, input_dir2=input_dir2, knn_k=knn_k, adj_mode=adj_mode, load_complete_graph=load_complete_graph)
    # Split into train and validation
    if n_folders == 1:
        train_data, valid_data = random_split(data, [n_train, n_valid])
    if n_folders == 2:
        train_data, valid_data = random_split(data, [2*n_train, 2*n_valid])
    return train_data, valid_data
