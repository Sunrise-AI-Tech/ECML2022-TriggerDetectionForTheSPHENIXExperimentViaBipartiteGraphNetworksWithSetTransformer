import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vp_GNN import GNN
from models.encoders import SoftPoolingGcnEncoder
from .utils import make_mlp

class Diffpool(nn.Module):
    def __init__(self, hidden_dim, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95, diff_pool_config={}):
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
        super(Diffpool, self).__init__()

        self.name = 'Diffpool'

        #input model
        self.input_network = make_mlp(3, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)

        self.loss_func = nn.BCELoss()

        # ip pred diffpool model
        self.ip_pred_diffpool = SoftPoolingGcnEncoder(input_dim=hidden_dim, **diff_pool_config)

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
                
    def forward(self, x, edge_index, batch, batch_size):
        node_output = self.input_network(x)
        n_hits = x.shape[0] // batch_size
        nodes = node_output.view(batch_size.item(), n_hits, node_output.shape[1])
        A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        start, end = edge_index
        A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = 1
        ip_pred = self.ip_pred_diffpool(nodes, A, batch_num_nodes=None)
        return ip_pred

    def train_model(self, ip_pred, ip_true):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true):
        sigmoid = nn.Sigmoid()
        return self.loss_func(sigmoid(ip_pred), ip_true)