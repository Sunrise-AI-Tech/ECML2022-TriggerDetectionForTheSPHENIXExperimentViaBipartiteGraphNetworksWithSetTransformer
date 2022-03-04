import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter


class PsiSuffix(nn.Module):
    def __init__(self, features, predict_diagonal):
        super().__init__()
        layers = []
        for i in range(len(features) - 2):
            layers.append(DiagOffdiagMLP(features[i], features[i + 1], predict_diagonal))
            layers.append(nn.ReLU())
        layers.append(DiagOffdiagMLP(features[-2], features[-1], predict_diagonal))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiagOffdiagMLP(nn.Module):
    def __init__(self, in_features, out_features, seperate_diag):
        super(DiagOffdiagMLP, self).__init__()

        self.seperate_diag = seperate_diag
        self.conv_offdiag = nn.Conv2d(in_features, out_features, 1)
        if self.seperate_diag:
            self.conv_diag = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x):
        # Assume x.shape == (B, C, N, N)
        if self.seperate_diag:
            return self.conv_offdiag(x) + (self.conv_diag(x.diagonal(dim1=2, dim2=3))).diag_embed(dim1=2, dim2=3)
        return self.conv_offdiag(x)


class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = small_in_features

        self.query = nn.Sequential(
            nn.Linear(in_features, small_in_features),
            nn.Tanh(),
        )
        self.key = nn.Linear(in_features, small_in_features)

    def forward(self, inp):
        # inp.shape should be (B,N,C)
        q = self.query(inp)  # (B,N,C/10)
        k = self.key(inp)  # B,N,C/10

        x = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # B,N,N

        x = x.transpose(1, 2)  # (B,N,N)
        x = x.softmax(dim=2)  # over rows
        x = torch.matmul(x, inp)  # (B, N, C)
        return x

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.tensor.new_zeros(x.size(0))
        
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        
        score = self.score_layer(x,edge_index).squeeze()
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm