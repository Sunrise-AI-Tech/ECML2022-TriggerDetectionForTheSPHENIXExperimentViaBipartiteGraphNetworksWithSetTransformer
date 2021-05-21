import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vp_GNN import GNN
from models.encoders import SoftPoolingGcnEncoder

class GNNDiffpool(nn.Module):
    def __init__(self, hidden_dim, input_dim=4, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95, diff_pool_config={}, mid_loss=False, mid_loss_weight=None):
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
        super(GNNDiffpool, self).__init__()

        if input_dim < 5:
            self.name = 'GNNDiffpool'
        else:
            self.name = 'GNN_Diffpool_trackinfo'

        # GNN model
        self.gnn = GNN(input_dim, **GNN_config)
        GNN_hidden_dim = GNN_config['hidden_dim']

        self.loss_func = nn.BCELoss()

        # ip pred diffpool model
        self.ip_pred_diffpool = SoftPoolingGcnEncoder(input_dim=GNN_hidden_dim, **diff_pool_config)

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)

        # mid_loss
        self.mid_loss = mid_loss
        self.mid_loss_weight = mid_loss_weight
                
    def forward(self, x, edge_index, batch, batch_size):
        edge_output, node_output, ip_summary = self.gnn(x, edge_index, batch)
        n_hits = x.shape[0] // batch_size
        nodes = node_output.view(batch_size.item(), n_hits, node_output.shape[1])
        A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        start, end = edge_index
        A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = edge_output
        ip_pred = self.ip_pred_diffpool(nodes, A, batch_num_nodes=None)
        return ip_pred

    def train_model(self, ip_pred, ip_true, mid_pred=None, mid_true=None):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true, mid_pred, mid_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true, mid_pred=None, mid_true=None):
        sigmoid = nn.Sigmoid()
        loss = self.loss_func(sigmoid(ip_pred), ip_true)
        if self.mid_loss:
            CELoss = nn.CrossEntropyLoss()
            # print(mid_pred.shape, mid_true.shape)
            loss += self.mid_loss_weight * CELoss(mid_pred, mid_true)
        return loss