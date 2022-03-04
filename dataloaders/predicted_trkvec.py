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
import logging

from numpy.linalg import inv
from icecream import ic

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
"""
    res['n_tracks'] = np.array(track_vector.shape[0])
    res['tracks_info'] = track_vector
    res['original_track_label'] = np.array([dict_list_psv[tuple(itm)] for itm in list_psv])
    res['pid'] = np.array(list_pid)
    res['momentum'] = np.array([momentum_dict[k] for k in list_pid])
    res['trigger_flag'] = np.array(trigger_flag)
    res['track_2nd_vertex'] = np.array(list_psv)
    res['modified_2nd_vertex'] = res['track_2nd_vertex']
    res['ip'] = ip

    if save:
        out_filename = os.path.join(dir_output, in_filename.split('/')[-1])
        np.savez(out_filename, **res) 
        
    res['tracks_to_hits'] = res_continent
    res['pid'] = pid
    
    is_trigger_track = []
    if track_vector.shape[0] != 0:
        is_trigger_track = (np.isclose(res['track_2nd_vertex'], ip, atol=0.001).all(axis=1)) == False
    res['is_trigger_track'] = is_trigger_track
    
    # calculate radius
    hits = track_vector[:, :15].reshape(track_vector.shape[0], 5, 3)
    good_hits = np.all(hits != 0, axis=-1)
    n_hits = np.sum(good_hits, axis=-1)
    r = get_approximate_radii(track_vector, n_hits, good_hits)
    res['r'] = r
    
    res['n_hits'] = n_hits

"""
def load_graph(filename, load_compelete_graph=False):
    with np.load(filename) as f:
        complete_flags = f['is_complete'] 
        if load_compelete_graph and len(complete_flags)!=0:
            track_vector = f['tracks_info'][complete_flags]
            origin_vertices = f['track_2nd_vertex'][complete_flags]
            momentums = f['momentum'][complete_flags]
            pids = f['pid'] #[complete_flags]
            radius = f['r'][complete_flags]
            is_trigger_track = f['is_trigger_track'][complete_flags]
            # ptypes = f ['ptypes'][complete_flags]
            # energy = f['energy'][complete_flags]
            energy = 0
        else:
            track_vector = f['tracks_info']
            origin_vertices = f['track_2nd_vertex']
            momentums = f['momentum']
            pids = f['pid']
            radius = f['r']
            is_trigger_track = f['is_trigger_track']
            # ptypes = f ['ptypes']
            # energy = f['energy']
            energy = 0
        trigger = f['trigger_flag']
        ip = f['ip']
        n_tracks = f['n_tracks']
        n_hits = f['n_hits']
        if n_tracks != 0:
            adj = (origin_vertices[:,None] == origin_vertices).all(axis=2)
        else:
            adj = None
    # need:
    # origin_vertices
    # adj
    # n_hits is number of hits for each track, it's a vector
    return track_vector, complete_flags, origin_vertices, momentums, pids, energy, trigger, ip, adj, radius, is_trigger_track, n_hits, n_tracks

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))

class TrkDataset():
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None,
                n_input_dir=1, input_dir2=None, input_dir3=None, random_permutation=True, n_samples2=None,
                n_samples3=None, load_complete_graph=False, corruption_level=0.0, use_radius=False, first_file_index=0):
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist))
            filenames = self.metadata.file.values
        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            # random.shuffle(filenames)
            self.filenames = filenames if n_samples is None else filenames[first_file_index:first_file_index+n_samples]
            if n_input_dir == 2:
                input_dir2 = os.path.expandvars(input_dir2)
                filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
                self.filenames += (filenames if n_samples2 is None else filenames[first_file_index:first_file_index+n_samples2])
            if n_input_dir == 3:
                input_dir2 = os.path.expandvars(input_dir2)
                filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
                # random.shuffle(filenames)
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
        self.n_hits = []

        for file_index in range(len(self.filenames)):
            # removed ptype
            track_vector, complete_flags, origin_vertices, momentums, pids, energy, trigger, ip, adj, radius, is_trigger_track, n_hits, n_tracks = load_graph(self.filenames[file_index], load_complete_graph)
            
            self.track_vector.append(track_vector)
            self.r.append(radius)
            self.n_hits.append(n_hits)

            n_tracks = track_vector.shape[0]
            if n_tracks != 0:
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
            self.is_trigger_track.append(is_trigger_track)
            # self.ip.append(ip)
        self.n_tracks = np.array(self.n_tracks)

    def __getitem__(self, index):
        return self.track_vector[index], self.origin_vertices[index], self.adjs[index], self.n_tracks[index], self.trigger[index], self.energy[index], self.momentums[index], self.is_trigger_track[index], self.r[index] #, self.n_hits[index]

    def __len__(self):
        return len(self.n_tracks)


def get_data_loaders(name, batch_size, **data_args):
    if name == 'trkvec_predicted':
        train_dataset, valid_dataset, train_data_n_nodes, valid_data_n_nodes = get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
    
    from torch_geometric.data import Batch
    # collate_fn = Batch.from_data_list
    train_batch_sampler = JetsBatchSampler(train_data_n_nodes, batch_size)
    train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    logging.info(f'For Training Dataset, {train_batch_sampler.drop_event_count} events (n_tracks <= 1) was removed; ') #{sum(train_dataset.y[train_batch_sampler.drop_event_index])} of them are non-trigger events.')
    valid_batch_sampler = JetsBatchSampler(valid_data_n_nodes, batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler)
    logging.info(f'For Validation Dataset, {valid_batch_sampler.drop_event_count} events (n_tracks <= 1) was removed;') # {sum(valid_dataset.y[valid_batch_sampler.drop_event_index])} of them are non-trigger events.')
    
    return train_data_loader, valid_data_loader


def get_datasets(n_train, n_valid, input_dir=None, filelist=None,
                n_folders=1, input_dir2=None, input_dir3=None, random_permutation=True,
                n_train2=None, n_valid2=None, n_train3=0, n_valid3=0, load_complete_graph=False, corruption_level=0.0, use_radius=False):
    n_train2 = n_train
    n_valid2 = n_valid
    data = TrkDataset(input_dir=input_dir, filelist=filelist,
                           n_samples=n_train+n_valid,
                           n_input_dir=n_folders, input_dir2=input_dir2, 
                           input_dir3=input_dir3, random_permutation=random_permutation,
                           n_samples2=n_train2+n_valid2, n_samples3=n_train3+n_valid3, load_complete_graph=load_complete_graph,\
                           corruption_level=corruption_level, use_radius=use_radius)

    # valid_rate = (n_valid+n_valid2) / (n_train+n_train2 + n_valid+n_valid2) 
    # total_events = len(data) 
    # valid_events = int(total_events * valid_rate)
    # train_events = total_events - valid_events
    # Split into train and validation
    if n_folders == 2:
        train_data, valid_data = random_split(data, [n_train+n_train2, n_valid+n_valid2])
        # train_data, valid_data = random_split(data, [train_events, valid_events])
    else:
        train_data, valid_data = random_split(data, [n_train, n_valid])

    train_data_n_nodes = data.n_tracks[train_data.indices]
    valid_data_n_nodes = data.n_tracks[valid_data.indices]
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
        self.drop_event_count = sum(n_nodes_array<=1)
        self.drop_event_index = np.where(n_nodes_array <= 1)[0]

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

