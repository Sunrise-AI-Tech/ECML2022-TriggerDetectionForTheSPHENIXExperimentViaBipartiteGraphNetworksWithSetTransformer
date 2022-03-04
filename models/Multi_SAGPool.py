import torch
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from models.layers import SAGPool

import torch_geometric
import icecream as ic

class SingelSAGPoolNet(torch.nn.Module):
    def __init__(self, is_hierarchical, num_features, nhid, num_classes, pooling_ratio, dropout_ratio, use_edge_attr):
        super(SingelSAGPoolNet, self).__init__()

        # self.args = args
        # self._num_features = num_features
        # self._nhid = nhid
        # self._num_classes = num_classes
        # self._pooling_ratio = pooling_ratio
        self._dropout_ratio = dropout_ratio
        self._is_hierarchical = is_hierarchical
        
        self.conv1 = GCNConv(num_features, nhid)
        self.pool1 = SAGPool(nhid, ratio=pooling_ratio)
        self.conv2 = GCNConv(nhid, nhid)
        self.pool2 = SAGPool(nhid, ratio=pooling_ratio)
        self.conv3 = GCNConv(nhid, nhid)
        self.pool3 = SAGPool(nhid, ratio=pooling_ratio)

        self.conv_global = GCNConv(nhid, nhid)
        self.pool_global = SAGPool(nhid*3, ratio=pooling_ratio)
        self.lin1_global = torch.nn.Linear(nhid*6, nhid)

        self.lin1 = torch.nn.Linear(nhid*2, nhid)
        # self.lin2 = torch.nn.Linear(nhid, nhid//2)
        # self.lin3 = torch.nn.Linear(nhid//2, num_classes)
        self.use_edge_attr = use_edge_attr

        # print("is_hierarchical: " + str(is_hierarchical))


    def forward(self, data, contrast=False):
        x, edge_index, batch, batch_size, edge_attr = data.x, data.edge_index, data.batch, data.batch_size, data.edge_attr
        edge_attr = data.edge_attr if self.use_edge_attr else None

        if self._is_hierarchical:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x, edge_index, edge_attr, batch, perm = self.pool1(x, edge_index, edge_attr, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x, edge_index, edge_attr, batch, perm = self.pool2(x, edge_index, edge_attr, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
            x = F.relu(self.conv3(x, edge_index, edge_attr))
            x, edge_index, edge_attr, batch, perm = self.pool3(x, edge_index, edge_attr, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
        else:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x1 = x

            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x2 = x
        
            x = F.relu(self.conv3(x, edge_index, edge_attr))
            x = torch.cat([x1, x2, x], dim = 1)

            x, edge_index, edge_attr, batch, perm = self.pool_global(x, edge_index, edge_attr, batch)
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.lin1_global(x))
        
        # x = F.dropout(x, p=self._dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class MultiSAGPoolNet(torch.nn.Module):
    def __init__(self, is_hierarchical, num_features, nhid, num_classes, pooling_ratio, dropout_ratio, use_edge_attr, n_head=2):
        super(MultiSAGPoolNet, self).__init__()

        # self.args = args
        # self._num_features = num_features
        # self._nhid = nhid
        # self._num_classes = num_classes
        # self._pooling_ratio = pooling_ratio
        self._dropout_ratio = dropout_ratio
        self._is_hierarchical = is_hierarchical
        self.n_head = n_head
        self.nhid_s = nhid//n_head
        
        self.SagPools = torch.nn.ModuleList([SingelSAGPoolNet(is_hierarchical, num_features, nhid//n_head, num_classes, pooling_ratio, dropout_ratio, use_edge_attr) for _ in range(n_head)])

        self.lin2 = torch.nn.Linear(self.nhid_s * self.n_head, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_classes)
        # self.use_edge_attr = use_edge_attr

        print("is_hierarchical: " + str(is_hierarchical) + ' n_head: '+ str(n_head))

    def forward(self, data, contrast=False):
        
        x = torch.cat([self.SagPools[i](data) for i in range(self.n_head)], dim=1)
        x = F.dropout(x, p=self._dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x