import torch
import torch.nn as nn
import torch.nn.functional as F

#from models.vp_GNN import GNN
from models.encoders import SoftPoolingGcnEncoder
from .utils import make_mlp
import logging

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.activation = getattr(nn, hidden_activation)
        self.network = nn.Sequential(
            nn.Conv2d(input_dim*3+1, hidden_dim, 1),
            # nn.GroupNorm(1, hidden_dim),
            self.activation(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            # nn.GroupNorm(1, hidden_dim),
            self.activation(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            # nn.GroupNorm(1, hidden_dim),
            self.activation(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            # nn.GroupNorm(1, hidden_dim),
            self.activation(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x, A, vp):
        x = x.transpose(2, 1)  # from BxNxC to BxCxN
        n = x.shape[2]
        m1 = x.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to rows  
        m2 = x.unsqueeze(3).repeat(1, 1, 1, n)  # broadcast to cols  
        m3 = vp.unsqueeze(2).unsqueeze(2).repeat(1, 1, n, n)
        block = torch.cat((m1, m2, m3, A.unsqueeze(1)), dim=1)
        return self.network(block).squeeze(1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*4, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, A, vp):
        # Aggregate edge-weighted incoming/outgoing features
        mi = torch.matmul(torch.transpose(A, 1, 2), x)
        mo = torch.matmul(A, x)
        node_inputs = torch.cat([mi, mo, x, vp.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=2)
        return self.network(node_inputs)

class GNN(nn.Module):
    """
    Dense GNN output node features and edge probabilities(for all connections).
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation='Tanh', layer_norm=True, batch_size=500):
        super(GNN, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        
        self.vp_network = make_mlp(hidden_dim, [hidden_dim] * 3,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, A):
        """Apply forward pass of the model"""

        # Apply input network to get hidden representation
        # logging.debug(f'input x size: {x.shape}')
        x = self.input_network(x)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vp = self.vp_network(torch.mean(x, dim=1))
        vp_all = vp
        
        # logging.debug(f'shape of vp: {vp.shape}')
        # logging.debug(f'x size after input network: {x.shape}')

        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x

            # Apply node network
            x = self.node_network(x, A, vp)

            # Apply edge network
            A = torch.sigmoid(self.edge_network(x, A, vp))

            # Residual connection
            vp = self.vp_network(torch.mean(x, dim=1))
            vp_all = torch.cat([vp_all, vp], dim=1)
            x = x + x0

        return x, A

class DenseGNNDiffpool(nn.Module):
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
        super(DenseGNNDiffpool, self).__init__()

        self.name = 'DenseGNNDiffpool'

        # GNN model
        self.gnn = GNN(input_dim=3, **GNN_config)
        GNN_hidden_dim = GNN_config['hidden_dim']

        self.loss_func = nn.BCELoss()

        # ip pred diffpool model
        self.ip_pred_diffpool = SoftPoolingGcnEncoder(input_dim=GNN_hidden_dim, **diff_pool_config)

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
                
    def forward(self, x, edge_index, batch, batch_size):
        n_hits = x.shape[0] // batch_size
        x = x.view(batch_size.item(), n_hits, x.shape[1])
        A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        start, end = edge_index
        A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = 1
        nodes, A = self.gnn(x, A)
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