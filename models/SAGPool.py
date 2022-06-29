import torch
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from models.layers import SAGPool

import torch_geometric
import icecream as ic

class SAGPoolNet(torch.nn.Module):
    def __init__(self, is_hierarchical, num_features, nhid, num_classes, pooling_ratio, dropout_ratio, use_edge_attr):
        super(SAGPoolNet, self).__init__()

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
        self.lin2 = torch.nn.Linear(nhid, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_classes)
        self.use_edge_attr = use_edge_attr

        print("is_hierarchical: " + str(is_hierarchical))

    def augment(self, data, trigger_tracks=None):
        aug1, aug2 = self.augmentor
        if trigger_tracks is not None:
            for aug in self.augmentor:
                if hasattr(aug, 'set_trigger_tracks'):
                    aug.set_trigger_tracks(trigger_tracks)

        x1, edge_index1, edge_weight1 = aug1(data.x, data.edge_index, data.edge_attr)
        x2, edge_index2, edge_weight2 = aug2(data.x, data.edge_index, data.edge_attr)
        g1 = torch_geometric.data.Data(x=x1, edge_index=edge_index1, y=data.y,
            batch=data.batch,
            batch_size=data.batch_size,
            edge_attr=edge_weight1).cuda()
        g2 = torch_geometric.data.Data(x=x2, edge_index=edge_index2, y=data.y,
            batch=data.batch,
            batch_size=data.batch_size,
            edge_attr=edge_weight2).cuda()

        return g1, g2


    def forward(self, data, contrast=False):
        x, edge_index, batch, batch_size, edge_attr = data.x, data.edge_index, data.batch, data.batch_size, data.edge_attr
        edge_attr = data.edge_attr if self.use_edge_attr else None

        if self.use_edge_attr:
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

            if contrast:
                return x
            
            x = F.dropout(x, p=self._dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.log_softmax(self.lin3(x), dim=-1)

            return x
        else:
            if self._is_hierarchical:
                x = F.relu(self.conv1(x, edge_index))
                x, edge_index, edge_attr, batch, perm = self.pool1(x, edge_index, edge_attr, batch)
                x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

                x = F.relu(self.conv2(x, edge_index))
                x, edge_index, edge_attr, batch, perm = self.pool2(x, edge_index, edge_attr, batch)
                x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            
                x = F.relu(self.conv3(x, edge_index))
                x, edge_index, edge_attr, batch, perm = self.pool3(x, edge_index, edge_attr, batch)
                x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
                x = x1 + x2 + x3

                x = F.relu(self.lin1(x))
            else:
                x = F.relu(self.conv1(x, edge_index))
                x1 = x

                x = F.relu(self.conv2(x, edge_index))
                x2 = x
            
                x = F.relu(self.conv3(x, edge_index))
                x = torch.cat([x1, x2, x], dim = 1)

                x, edge_index, edge_attr, batch, perm = self.pool_global(x, edge_index, edge_attr, batch)
                x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

                x = F.relu(self.lin1_global(x))

            if contrast:
                return x
            
            x = F.dropout(x, p=self._dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.log_softmax(self.lin3(x), dim=-1)

            return x

    def predict(x):
        x = F.dropout(x, p=self._dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
