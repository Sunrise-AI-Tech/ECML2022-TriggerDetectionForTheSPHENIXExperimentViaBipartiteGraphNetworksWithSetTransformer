import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vp_GNN import GNN
from models.encoders import SoftPoolingGcnEncoder
from models.utils import make_mlp
from torch_scatter import scatter_add

class Pairwise_Predictor(nn.Module):
    def __init__(self, in_features, hid_dim, hidden_activation='Tanh', layer_norm=True):
        """
        SetToGraph_and_GLap model.
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(Pairwise_Predictor, self).__init__()
        # MLP, input: phi(x) output: coordinate with same size
        if hidden_activation == 'Tanh':
            self.activation = nn.Tanh()
        elif hidden_activation == 'ReLU':
            self.activation = nn.ReLU()
        self.mlp_affinity = nn.Sequential(
            self.activation,
            nn.Conv2d(in_features*2, hid_dim, 1),
            self.activation,
            # nn.Conv2d(hid_dim, hid_dim, 1),
            # self.activation,
            # nn.Conv2d(hid_dim, hid_dim, 1),
            # self.activation,
            # nn.Conv2d(hid_dim, hid_dim, 1),
            # self.activation,
            nn.Conv2d(hid_dim, 1, 1),

        )
        # self.mlp_coord=make_mlp(in_features, [hid_dim, hid_dim, 3],
        #                               hidden_activation,
        #                               output_activation=None,
        #                               layer_norm=layer_norm)
    def forward(self, x):
        x = x.transpose(2, 1)  # from BxNxC to BxCxN

        n = x.shape[2]
        # broadcast
        m1 = x.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to rows  
        m2 = x.unsqueeze(3).repeat(1, 1, 1, n)  # broadcast to cols  
        block = torch.cat((m1, m2), dim=1) 

        # print(f'block: {block}')

        # predict affinity and sv
        edge_vals = self.mlp_affinity(block) # shape (B,out_features,N,N)
        # cds=self.mlp_coord(x.transpose(2, 1))
        edge_vals = edge_vals.squeeze(1)
        edge_vals = edge_vals + edge_vals.transpose(1, 2)
        return edge_vals

class GNNPairDiffpool(nn.Module):
    def __init__(self, hidden_dim, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95, diff_pool_config={}, affinity_loss=False, affinity_loss_CE_weight=0.1, affinity_loss_Lp_weight=0.1):
        """
        SetToGraph model.
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(GNNPairDiffpool, self).__init__()
        
        self.name = 'GNNPairDiffpool'

        # GNN model
        self.gnn = GNN(input_dim=3, **GNN_config)
        GNN_hidden_dim = GNN_config['hidden_dim']

        # Pairwise Predictor
        self.pariwise_predictor = Pairwise_Predictor(GNN_hidden_dim, hidden_dim, hidden_activation, layer_norm)

        # ip pred diffpool model
        self.ip_pred_diffpool = SoftPoolingGcnEncoder(input_dim=GNN_hidden_dim, **diff_pool_config)

        # loss function
        self.loss_func = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
        # affinity loss
        self.affinity_loss = affinity_loss
        self.affinity_loss_CE_weight = affinity_loss_CE_weight
        self.affinity_loss_Lp_weight = affinity_loss_Lp_weight
                
    def forward(self, x, edge_index, batch, batch_size, hits_to_track, hits_cumsum, n_tracks):
        edge_output, node_output, ip_summary = self.gnn(x, edge_index, batch)

        # n_hits = x.shape[0] // batch_size
        # print(batch_size.item(), n_hits.item(), node_output.shape[1])
        # nodes = node_output.view(batch_size.item(), n_hits, node_output.shape[1])
        # A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        # start, end = edge_index
        # A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = edge_output
        # ip_pred = self.ip_pred_diffpool(nodes, A, batch_num_nodes=None)
        
        start = 0
        track_summaries = []
        for end in hits_cumsum:
            hits = node_output[start:end]
            h2t = hits_to_track[start:end]
            track_summaries.append(scatter_add(hits, h2t, dim=0, dim_size=n_tracks))
            start = end
        track_summaries = torch.stack(track_summaries)

        edge_vals = self.pariwise_predictor(track_summaries)
        A = self.sigmoid(edge_vals)
        # print(track_summaries)
        # print(track_summaries.shape, A.shape)
        ip_pred = self.ip_pred_diffpool(track_summaries, A, batch_num_nodes=None)
        
        # print(ip_pred)
        return ip_pred, A

    def train_model(self, ip_pred, ip_true, A_pred=None, A_true=None, track_vtx=None):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true, A_pred, A_true, track_vtx)
        loss.backward()
        # for n, p in self.named_parameters():
        #     if p.grad is not None:
        #         print(n, p.grad.abs().max())
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true, A_pred=None, A_true=None, track_vtx=None):
        loss = self.loss_func(self.sigmoid(ip_pred), ip_true)
        if self.affinity_loss:
            loss += self.affinity_loss_CE_weight*self.loss_func(A_pred, A_true) 
            loss += self.affinity_loss_Lp_weight*self.get_normalized_laplacian_loss(track_vtx, A_pred)
        return loss

    def get_normalized_laplacian_loss(self, cd_hat,A):
        v=torch.sum(A,dim=1)
        D=torch.diag_embed(v)
        Glap=D-A
        normalized_Glap = torch.bmm(torch.bmm(torch.sqrt(D), Glap), torch.sqrt(D))
        ans=torch.bmm(torch.bmm(cd_hat.transpose(2,1),normalized_Glap),cd_hat) # C^T * Graph laplacian * C
        loss=torch.sum(torch.diagonal(ans,dim1=1,dim2=2))
        return loss