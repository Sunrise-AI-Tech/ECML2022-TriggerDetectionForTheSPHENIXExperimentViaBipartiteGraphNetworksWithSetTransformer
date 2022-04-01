from os import X_OK
from sklearn import utils
from torch import tensor
from torch._C import is_anomaly_enabled
import torch.nn as nn
import torch
from typing import OrderedDict, Tuple
from icecream import ic
from torch.nn.modules import distance
from torch_scatter import scatter_add
# from utils import isParallelMatrix, shortest_dist_parallel_matrix, shortest_dist_non_parallel_matrix

from random import sample
import logging
from icecream import ic


class GarNet(nn.Module):
    def __init__(
            self,
            num_features,
            layers_spec, # Tuple of (N, feature_dim, coordinate_dim)
            num_classes,
            hidden_activation='Tanh', 
            aggregator_activation='potential'
            ):
        super(GarNet, self).__init__()
        garnet_layers = []
        prev_dim = num_features
        
        for feature_dim, n_aggregators in layers_spec:
            garnet_layers.append(
                    GarNetLayer(
                        prev_dim,
                        feature_dim,
                        n_aggregators,
                        hidden_activation,
                        aggregator_activation,
                    )
            )
            prev_dim = feature_dim 

        self._layers = nn.Sequential(
                *garnet_layers
        )

        self._pred_layers = nn.Sequential(
                nn.Linear(2*prev_dim, prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, num_classes)
        )

    def forward(self, X):
        pred_x = self._layers(X)
        mean_pooled = torch.mean(pred_x, axis=-2)
        max_pooled = torch.max(pred_x, axis=-2)[1]
        H = torch.cat((mean_pooled, max_pooled), axis=-1)

        return self._pred_layers(H)

class GarNetLayer(nn.Module):
    def __init__(self,
        input_dim,
        feature_dim,
        n_aggregators,
        hidden_activation,
        aggregator_activation
    ):
        super(GarNetLayer, self).__init__()
        self.aggregator_activation = aggregator_activation
        self.transform_in = nn.Linear(input_dim, feature_dim + n_aggregators)
        self.transform_out = nn.Sequential(
                nn.Linear(2*feature_dim*n_aggregators + input_dim, feature_dim),
                getattr(nn, hidden_activation)()
        )
        self._feature_dim = feature_dim
        self._n_aggregators = n_aggregators

    def forward(self, X):
        """
        X: Tensor of shape [n_minibatches, n_tracks, n_track_features]
        """
        #        X = X.transpose(-1, -2)
        Xp = self.transform_in(X)
        F = Xp[..., :self._feature_dim]
        distances = Xp[..., self._feature_dim:]
        
        if self.aggregator_activation == 'potential':
            potential = torch.exp(-torch.abs(distances))
        elif self.aggregator_activation == 'ReLU':
            act = nn.ReLU()
            potential = act(distances)
        elif self.aggregator_activation == 'Tanh':
            act = nn.Tanh()
            potential = act(distances)
        else:
            potential = distances
        # potential is of shape (n_minibatches, n_tracks, n_aggregators)
        edges = torch.einsum('btf,bta->baft', F, potential)
        max_pooled = torch.max(edges, dim=-1)[0] # (n_minibatches, n_aggregators, n_features)
        # (n_minimatches, n_tracks, n_features)
        mean_pooled = torch.mean(edges, dim=-1)
        pooled = torch.cat((max_pooled, mean_pooled), axis=-1)
        pooled = pooled.reshape(pooled.shape[0], 1, pooled.shape[1]*pooled.shape[2]).repeat(1, X.shape[1], 1)
        H = torch.cat((X, pooled), axis=-1)
        return self.transform_out(H)
