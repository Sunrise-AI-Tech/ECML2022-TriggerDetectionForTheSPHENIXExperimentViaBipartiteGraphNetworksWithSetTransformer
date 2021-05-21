import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.deep_sets import DeepSet
# from models.layers import PsiSuffix
# from models.s2g_and_glap import SetToGraph_and_GLap
# from models.encoders import SoftPoolingGcnEncoder
from models.utils import make_mlp
# from models.utils import NodeNetwork, EdgeNetwork
# from models.GNN import GNN
from torch_scatter import scatter_add
# from models.pariwise_predictor import Pairwise_Predictor
# from models.encoders import SoftPoolingGcnEncoder

"""
This module implements the PyTorch modules that define the sparse
message-passing graph neural networks for segment classification.
In particular, this implementation utilizes the pytorch_geometric
and supporting libraries:
https://github.com/rusty1s/pytorch_geometric
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add

# Locals
from .utils import make_mlp

"""
This module implements the PyTorch modules that define the sparse
message-passing graph neural networks for segment classification.
In particular, this implementation utilizes the pytorch_geometric
and supporting libraries:
https://github.com/rusty1s/pytorch_geometric
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import logging
# Locals
from .utils import make_mlp

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
        self.network = make_mlp(input_dim*3,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index, vp, batch):
        # Select the features of the associated nodes
        start, end = edge_index
        #x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end], vp[batch[start]]], dim=1)
        return self.network(edge_inputs).squeeze(-1)
    

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

    def forward(self, x, e, vp, edge_index, batch):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x, vp[batch]], dim=1)
        return self.network(node_inputs)

class GNN(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
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

        self.ip_summary_network = make_mlp(hidden_dim*(n_graph_iters+1), [hidden_dim] * 3,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)
        
        # Set LSTM layer to combine all node information
        #self.combine_layer = nn.LSTM(hidden_dim, hidden_dim)
        
        # Setup the output layers
        #self.output_network = make_mlp(hidden_dim, [hidden_dim, hidden_dim, hidden_dim, 1], output_activation=hidden_activation, layer_norm=layer_norm)

    def forward(self, x, edge_index, batch):
        """Apply forward pass of the model"""

        # Apply input network to get hidden representation
        logging.debug(f'input x size: {x.shape}')
        x = self.input_network(x)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #temp = torch.cuda.FloatTensor(x[(inputs.layers==0) | (inputs.layers==1)].shape[0]).fill_(1)
        #mean_factor = scatter_add(temp, inputs.batch[(inputs.layers==0) | (inputs.layers==1)])[:, None]
        #vp = scatter_add(x[(inputs.layers==0) | (inputs.layers==1)], inputs.batch[(inputs.layers==0) | (inputs.layers==1)], dim=0)/mean_factor
        temp = torch.cuda.FloatTensor(x.shape[0]).fill_(1)
        mean_factor = scatter_add(temp, batch)[:, None]
        # print(batch.shape)
        # print(x.shape)
        # print(scatter_add(x, batch, dim=0).shape)
        # print(mean_factor.shape)
        vp = self.vp_network(scatter_add(x, batch, dim=0)/mean_factor)
        vp_all = vp
        
        logging.debug(f'shape of vp: {vp.shape}')
        logging.debug(f'x size after input network: {x.shape}')
        # Shortcut connect the inputs onto the hidden representation
        #x = torch.cat([x, inputs.x], dim=-1)

        # initalize global feature with random value
        #global_feature = torch.randn(self.hidden_dim, device=torch.device('cuda:0'))
        #logging.debug(f'global feature shape after initalization {global_feature.shape}')
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x

            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index, vp, batch))
            logging.debug(f'shape after edge network: {e.shape}')

            # Apply node network
            x = self.node_network(x, e, vp, edge_index, batch)
            logging.debug(f'shape after node network: {x.shape}') 
            # Shortcut connect the inputs onto the hidden representation
            #x = torch.cat([x, inputs.x], dim=-1)

            # Residual connection
            #vp = vp + scatter_add(x[(inputs.layers==0) | (inputs.layers==1)], inputs.batch[[(inputs.layers==0) | (inputs.layers==1)]], dim=0)/mean_factor
            vp = self.vp_network(scatter_add(x, batch, dim=0)/mean_factor)
            vp_all = torch.cat([vp_all, vp], dim=1)
            x = x + x0

        # use LSTM to combine all node information into one
        #full_node, (global_node, cn) = self.combine_layer(x.view(x.shape[0],1,-1))
        #logging.debug(f'shape after LSTM: {global_node.shape}')
            
            #global_node = global_node[:, global_node.shape[-2]-1, :].squeeze()
            #logging.d ebug(f'shape after slice: {global_node.shape}')

            # feed global node and previous global feature to update global feature
            #logging.debug(f'shape of global feature; {global_feature.shape}')
            #logging.debug(f'shape of the cat tensor: {torch.cat([global_feature, global_node]).shape}')
        
        ip_summary = self.ip_summary_network(vp_all)
        logging.debug(f'shape of ip: {ip_summary.shape}')
        return e, x, ip_summary



# class VpGNN(nn.Module):
#     def __init__(self, hidden_dim, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95, loss_weights={}):
#         """
#         SetToGraph model.
#         :param in_features: input set's number of features per data point
#         :param out_features: number of output features.
#         :param set_fn_feats: list of number of features for the output of each deepsets layer
#         :param method: transformer method - quad, lin2 or lin5
#         :param hidden_mlp: list[int], number of features in hidden layers mlp.
#         :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
#         :param attention: Bool. Use attention in DeepSets
#         :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
#         """
#         super(VpGNN, self).__init__()
#         self.name = 'VpGNN'

#         # GNN model
#         self.gnn = GNN(input_dim=3, **GNN_config)
#         GNN_hidden_dim = GNN_config['hidden_dim']

#         # ip pred
#         self.ip_pred_mlp = make_mlp(GNN_hidden_dim, [hidden_dim]*3+[3],
#                                 hidden_activation=hidden_activation,
#                                 output_activation=None,
#                                 layer_norm=layer_norm)

#         self.loss_func = nn.MSELoss()

#         # pairwise model
#         # self.pariwise_predictor = Pairwise_Predictor(GNN_hidden_dim, hidden_dim, hidden_activation, layer_norm)
#         # self.set2graph = SetToGraph_and_GLap(GNN_hidden_dim, 1, \
#         #      set_fn_feats=[hidden_dim, hidden_dim, hidden_dim, hidden_dim, 5], method='lin2',\
#         #      hidden_mlp=[hidden_dim], predict_diagonal=False, attention=True, cfg=None, dropout_rate=0)
        
#         # # diffpool model
#         # self.diffpool = SoftPoolingGcnEncoder(input_dim=hidden_dim, **diff_pool_config, bn=True)

#         # optimizer init
#         self.learning_rate = learning_rate
#         self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
#         # self.optimizer_s2g = torch.optim.Adam(params= list(self.input_network.parameters()) + list(self.node_network.parameters()) + list(self.set2graph.parameters()), lr=learning_rate)
#         # self.optimizer_diffpool = torch.optim.Adam(params=self.diffpool.parameters(), lr=learning_rate)

#         # lr_scheduler
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
#         # add the loss weight 
#         # self.wCE = loss_weights['weight_loss_CE']
#         # self.wMSE = loss_weights['weight_loss_MSE']
#         # self.wGlap = loss_weights['weight_loss_Glap']
#         # self.wLabel = loss_weights['weight_loss_trigger']

                
#     def forward(self, x, edge_index, batch, batch_size):
#         edge_output, node_output, ip_summary = self.gnn(x, edge_index, batch)
#         ip_summary = scatter_add(node_output, batch, dim=0, dim_size=batch_size)
#         ip_pred = self.ip_pred_mlp(ip_summary)
#         return ip_pred

#     def train_model(self, ip_pred, ip_true):
#         self.optimizer.zero_grad()
#         loss = self.get_loss(ip_pred, ip_true)
#         loss.backward()
#         self.optimizer.step()
#         return loss

#     def get_loss(self, ip_pred, ip_true):
#         return self.loss_func(ip_pred, ip_true)
