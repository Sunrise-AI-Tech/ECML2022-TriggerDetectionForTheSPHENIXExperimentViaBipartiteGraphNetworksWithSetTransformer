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
import tqdm

from numpy.linalg import inv

def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)

def get_approximate_radius(tracks_info, is_complete):
    complete_tracks = tracks_info[is_complete]
    # complete_track_momentums = track_momentum[is_complete]
    A = np.ones((complete_tracks.shape[0], 5, 3))
    A[:, :, 0] = complete_tracks[:, [0, 3, 6, 9, 12]]
    A[:, :, 1] = complete_tracks[:, [1, 4, 7, 10, 13]]
    y = - complete_tracks[:, [0, 3, 6, 9, 12]]**2 - complete_tracks[:, [1, 4, 7, 10, 13]]**2
    y = y.reshape((y.shape[0], y.shape[1], 1))
    # c = np.einsum('lij,ljk->lik', inv(A), y)
    AT = np.transpose(A, axes=(0, 2, 1))
    # print(A.shape, AT.shape, y.shape)
    # c = inv(matmul_3D(A, AT))
    c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
    # print(A.shape, AT.shape, y.shape, c.shape)
    r = np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
    return r

def get_approximate_radii(tracks_info, n_hits, good_hits):
    x_indices = [3*j for j in range(5)]
    y_indices = [3*j+1 for j in range(5)]
    r = np.zeros((tracks_info.shape[0], 1))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        hit_indices = good_hits[n_hits == n_hit]
        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_hit, 3))
        x_values = complete_tracks[:, x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)

        y_values = complete_tracks[:, y_indices]
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)
        A[:, :, 0] = x_values
        A[:, :, 1] = y_values

        y = - x_values**2 - y_values**2
        y = y.reshape((y.shape[0], y.shape[1], 1))
        AT = np.transpose(A, axes=(0, 2, 1))
        c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
        r[n_hits == n_hit] == 1
        r[n_hits == n_hit] = np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
    #test = get_approximate_radius(tracks_info, n_hits == 5)
    #assert np.allclose(test, r[n_hits == 5])

    return r


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
            adj = np.array([[]])
    # need:
    # origin_vertices
    # adj
    return track_vector, complete_flags, origin_vertices, momentums, pids, ptypes, energy, trigger, ip, adj

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))

class TrkDataset():
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None,
                n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True, n_samples2=None,
                n_samples3=None, load_complete_graph=False, add_geo_features=False, corruption_level=0.0, use_radius=False):
        assert use_radius
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist))
            filenames = self.metadata.file.values
        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            random.shuffle(filenames)
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
                random.shuffle(filenames)
                self.filenames += (filenames if n_samples2 is None else filenames[:(n_samples2)])

                input_dir3 = os.path.expandvars(input_dir3)
                filenames = sorted([os.path.join(input_dir3, f) for f in os.listdir(input_dir3)
                                if f.startswith('event')])
                self.filenames += (filenames if n_samples3 is None else filenames[:(n_samples3)])
            # random.shuffle(self.filenames)
        else:
            raise Exception('Must provide either input_dir or filelist to HitGraphDataset')
        
        self.random_permutation = random_permutation
        self.track_vector = []
        self.complete_flags = []
        self.origin_vertices = []
        self.momentums = []
        self.pids = []
        self.energy = []
        self.trigger = []
        self.n_tracks = []
        self.ptypes = []
        self.ip = []
        self.adjs = []
        self.is_trigger_track = []
        self.r = []

        for file_index in range(len(self.filenames)):
            track_vector, complete_flags, origin_vertices, momentums, pids, ptypes, energy, trigger, ip, adj = load_graph(self.filenames[file_index], load_complete_graph)
            if track_vector.shape[0] != 0 and add_geo_features:
                # 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 13
                geo_features = np.zeros((track_vector.shape[0], 13))
                phi  = np.zeros((track_vector.shape[0], 5))
                geo_features[:, 5] = np.arctan2(track_vector[:, 1], track_vector[:, 0])
                for i in range(4):
                    geo_features[:, i] = get_length(track_vector[:, (3*i+3):(3*i+6)], track_vector[:, (3*i):(3*i+3)])
                for i in range(5):
                    phi[:, i] = np.arctan2(track_vector[:, (3*i)+1], track_vector[:, (3*i)])
                geo_features[:, 5] = get_length(track_vector[:, 12:15], track_vector[:, 0:3])
                geo_features[:, 6:10] = np.diff(phi)
                geo_features[:, 10:13] = np.mean(track_vector.reshape((track_vector.shape[0], 5, 3)), axis=(0, 1))
                track_vector = np.concatenate([track_vector, geo_features], axis=1)
            self.track_vector.append(track_vector)

            if use_radius:
                hits = track_vector[:, :15].reshape(track_vector.shape[0], 5, 3)
                good_hits = np.all(hits != 0, axis=-1)
                n_hits = np.sum(good_hits, axis=-1)
                r = get_approximate_radii(track_vector, n_hits, good_hits)
                self.r.append(r)
            else:
                self.r.append(np.zeros((track_vector.shape[0], 1)))

            n_tracks = track_vector.shape[0]
            adj = torch.from_numpy(adj).view(n_tracks, n_tracks).to(torch.bool)

            if corruption_level > 0:
                corrupt = torch.rand((n_tracks, n_tracks)) > (1 - corruption_level)
                A = torch.triu(adj ^ corrupt)
                A = (A | A.T).fill_diagonal_(True).to(torch.float)
            else:
                A = adj

            self.adjs.append(A)


            # self.complete_flags.append(complete_flags)
            self.origin_vertices.append(origin_vertices)
            self.momentums.append(momentums)
            # self.pids.append(pids)
            # self.ptypes.append(ptypes)
            self.energy.append(energy)
            self.trigger.append(trigger)
            self.n_tracks.append(track_vector.shape[0])
            self.is_trigger_track.append(np.all(origin_vertices != ip, axis=-1))
            # self.ip.append(ip)
        self.n_tracks = np.array(self.n_tracks)

    def __getitem__(self, index):
        return self.track_vector[index], self.origin_vertices[index], self.adjs[index], self.n_tracks[index], self.trigger[index], self.energy[index], self.momentums[index], self.is_trigger_track[index], self.r[index]

    def __len__(self):
        return len(self.n_tracks)


def get_data_loaders(name, batch_size, **data_args):
    if name == 'trkvec-ecml':
        train_dataset, valid_dataset, test_dataset, train_data_n_nodes, valid_data_n_nodes, test_data_n_nodes = get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
    
    from torch_geometric.data import Batch
    # collate_fn = Batch.from_data_list
    train_batch_sampler = JetsBatchSampler(train_data_n_nodes, batch_size)
    train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    valid_batch_sampler = JetsBatchSampler(valid_data_n_nodes, batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler)
    test_batch_sampler = JetsBatchSampler(test_data_n_nodes, batch_size)
    test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)
    
    return train_data_loader, valid_data_loader, test_data_loader


def get_datasets(n_train, n_valid, n_test, 
        train_input_dir=None, valid_input_dir=None, test_input_dir=None,
                n_folders=1, input_dir2=None, input_dir3=None, random_permutation=True,
                n_train2=None, n_valid2=None, n_train3=0, n_valid3=0, load_complete_graph=False, add_geo_features=False, corruption_level=0.0, use_radius=True):
    n_train2 = n_train
    n_valid2 = n_valid
    train_data = TrkDataset(input_dir=train_input_dir, filelist=None,
                           n_samples=n_train,
                           n_input_dir=1, input_dir2=None, 
                           input_dir3=None, random_permutation=random_permutation,
                           n_samples2=0, n_samples3=0, load_complete_graph=load_complete_graph,\
                           add_geo_features=add_geo_features, corruption_level=corruption_level, use_radius=use_radius)

    valid_data = TrkDataset(input_dir=valid_input_dir, filelist=None,
                           n_samples=n_valid,
                           n_input_dir=1, input_dir2=None, 
                           input_dir3=None, random_permutation=random_permutation,
                           n_samples2=0, n_samples3=0, load_complete_graph=load_complete_graph,\
                           add_geo_features=add_geo_features, corruption_level=corruption_level, use_radius=use_radius)

    test_data = TrkDataset(input_dir=test_input_dir, filelist=None,
                           n_samples=n_test,
                           n_input_dir=1, input_dir2=None, 
                           input_dir3=None, random_permutation=random_permutation,
                           n_samples2=0, n_samples3=0, load_complete_graph=load_complete_graph,\
                           add_geo_features=add_geo_features, corruption_level=corruption_level, use_radius=use_radius)

    train_data_n_nodes = train_data.n_tracks
    valid_data_n_nodes = valid_data.n_tracks
    test_data_n_nodes = test_data.n_tracks
    return train_data, valid_data, test_data, train_data_n_nodes, valid_data_n_nodes, test_data_n_nodes

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

