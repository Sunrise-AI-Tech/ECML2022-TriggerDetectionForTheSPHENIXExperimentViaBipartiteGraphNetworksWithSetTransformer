import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.deep_sets import DeepSet
# from models.layers import PsiSuffix
# from models.s2g_and_glap import SetToGraph_and_GLap
# from models.encoders import SoftPoolingGcnEncoder
from models.utils import make_mlp
# from models.utils import NodeNetwork, EdgeNetwork
from models.GNN import GNN
from torch_scatter import scatter_add
# from models.pariwise_predictor import Pairwise_Predictor
# from models.encoders import SoftPoolingGcnEncoder

class IpGNN(nn.Module):
    def __init__(self, hidden_dim, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95, loss_weights={}):
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
        super(IpGNN, self).__init__()
        self.name = 'IpGNN'

        # GNN model
        self.gnn = GNN(input_dim=3, **GNN_config)
        GNN_hidden_dim = GNN_config['hidden_dim']

        # ip pred
        self.ip_pred_mlp = make_mlp(GNN_hidden_dim, [hidden_dim]*3+[3],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

        self.loss_func = nn.MSELoss()

        # pairwise model
        # self.pariwise_predictor = Pairwise_Predictor(GNN_hidden_dim, hidden_dim, hidden_activation, layer_norm)
        # self.set2graph = SetToGraph_and_GLap(GNN_hidden_dim, 1, \
        #      set_fn_feats=[hidden_dim, hidden_dim, hidden_dim, hidden_dim, 5], method='lin2',\
        #      hidden_mlp=[hidden_dim], predict_diagonal=False, attention=True, cfg=None, dropout_rate=0)
        
        # # diffpool model
        # self.diffpool = SoftPoolingGcnEncoder(input_dim=hidden_dim, **diff_pool_config, bn=True)

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        # self.optimizer_s2g = torch.optim.Adam(params= list(self.input_network.parameters()) + list(self.node_network.parameters()) + list(self.set2graph.parameters()), lr=learning_rate)
        # self.optimizer_diffpool = torch.optim.Adam(params=self.diffpool.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
        # add the loss weight 
        # self.wCE = loss_weights['weight_loss_CE']
        # self.wMSE = loss_weights['weight_loss_MSE']
        # self.wGlap = loss_weights['weight_loss_Glap']
        # self.wLabel = loss_weights['weight_loss_trigger']

                
    def forward(self, x, edge_index, batch, batch_size):
        edge_output, node_output = self.gnn(x, edge_index)
        ip_summary = scatter_add(node_output, batch, dim=0, dim_size=batch_size)
        ip_pred = self.ip_pred_mlp(ip_summary)
        return ip_pred
        # print(f'x.shape : {x.shape}')
        # print(f'hits_to_track: {hits_to_track}')
        # print(f'n_tracks: {n_tracks}')
        # print(f'x: {x}')
        # print(f'hits_to_track: {hits_to_track}')
        # start = 0
        # track_summaries = []
        # for end in hits_cumsum:
        #     hits = x[start:end]
        #     h2t = hits_to_track[start:end]
        #     # print(f'hits: {hits}')
        #     # print(f'h2t: {h2t}')
        #     track_summaries.append(scatter_add(hits, h2t, dim=0, dim_size=n_tracks))
        #     # print(f'track_summaries: {track_summaries}')
        #     start = end
        # track_summaries = torch.stack(track_summaries)
        # # print(f'track_summaries: {track_summaries}')
        # # print(track_summaries.shape)
        # # edge_vals, cds = self.set2graph(track_summaries)
        # edge_vals, cds = self.pariwise_predictor(track_summaries)
        
        # print(f'edge_vals: {edge_vals}')
        # print(f'cds: {cds}')
        # sigmoid = nn.Sigmoid()
        # A = sigmoid(edge_vals)
        # # relu = nn.ReLU()
        # # A = relu(A-self.relu_boundary)
        # # A = A + self.relu_boundary * (A>0)
        # pred_labels = self.diffpool(track_summaries, A[:, 0, :, :], batch_num_nodes=None)
        # # pred_labels = []
        # # x = u.transpose(2, 1)
        # # print(x[0].shape)
        # # sigmoid = nn.Sigmoid()
        # # A = sigmoid(edge_vals)
        # # for i in range(u.shape[0]):
        # #     pred_labels.append(self.diffpool(x[i], A[i, 0, :, :], batch_num_nodes=None))  
        # edge_vals = edge_vals.squeeze(1)
        # edge_vals = edge_vals + edge_vals.transpose(1, 2)
        # return 

    def train_model(self, ip_pred, ip_true):
        self.optimizer.zero_grad()
        loss = self.get_loss(ip_pred, ip_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true):
        return self.loss_func(ip_pred, ip_true)

