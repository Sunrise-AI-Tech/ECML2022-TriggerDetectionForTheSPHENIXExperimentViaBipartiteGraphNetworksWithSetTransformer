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

ActivationSpec = str
PoolingSpec = str
EdgeMLPSpec = Tuple[int]
PointSpec = Tuple[int]
EdgeConvSpec = Tuple[int, PointSpec, EdgeMLPSpec, PoolingSpec, ActivationSpec]
PredMLPSpec = Tuple[int, float]

def shortest_dist_non_parallel_matrix(v1,v2,p1,p2,i_dim): #v1, v2 is the 3D vector
    # method part 2, calculate the shortest distance by cross product
    v3=torch.cross(v1,v2, dim=i_dim) #the orthogonal vector
    v3_l=torch.linalg.norm(v3, dim=i_dim) #the length of v3
    dst=torch.sum(v3*torch.subtract(p2,p1), axis=i_dim) / v3_l
    
    return dst

def shortest_dist_parallel_matrix(v1,v2,p1,p2,i_dim):
    if v1.equal(v2):
        print("non-parallel")
    v_norm = torch.linalg.norm(v1, dim=i_dim)
    # ic(v_norm.shape)
    # ic(v1.shape)
    v = torch.div(v1.transpose(2,1).transpose(1,0), v_norm)
    v = v.transpose(1,0).transpose(2,1)
    # ic(v.shape)
    w = p2 - p1
    res = torch.linalg.norm(torch.cross(v,w, dim=i_dim), dim=i_dim)
    return(res)

def isParallelMatrix(a0, a1, b0, b1, i_dim):
    res = (torch.all(torch.cross(a0, a1, dim=i_dim) == 0, dim = i_dim, keepdim=True))
    return res

"""
Get distance between two line segments.

@Input:
    points: [n_minibatches, n_track_features, n_tracks]

@Output:
    distances: [n_minibatches, n_tracks, n_tracks]
"""

def get_distance(points):
    # ic(points.shape)
    points = points.transpose(1,2)
    p_0 = points[:, :, 0:3]
    p_2 = points[:, :, 3:6]

    # p_0 = p_0.transpose(2, 3)
    # p_2 = p_2.transpose(2, 3)
    # p_0 = p_0.transpose(1, 2)
    # p_2 = p_2.transpose(1, 2)
    # p_0 = p_0.transpose(0, 1)
    # p_2 = p_2.transpose(0, 1)

    # For a track there are 3 points: P0, P1, P2
    # Get line of track by using the P0 and P2
    # L(t) = V*t + P0 where V = P2 - P0
    # a is the slope, and b is the position
    try:
        a = p_2 - p_0
    except:
        ic(p_2)
        ic(p_0)
    b = p_0

    # Test
    # p = torch.Tensor([[[0,0,0,6,6,6,1,1,1], [0, 0, 1, 7,7,7, 0, 2, 0]]])

    # Get a0, a1, b0, b1
    [n_b, n_t, n_d] = a.size()

    a0 = a.repeat_interleave(n_t, dim = 1)
    a0 = a0.view(n_b, n_t * n_t, n_d)

    a1 = a.repeat_interleave(n_t, dim = 0)
    a1 = a1.view(n_b, n_t * n_t, n_d)

    b0 = b.repeat_interleave(n_t, dim = 1)
    b0 = b0.view(n_b, n_t * n_t, n_d)

    b1 = b.repeat_interleave(n_t, dim = 0)
    b1 = b1.view(n_b, n_t * n_t, n_d)

    [len_b, len_m, len_d] = a0.size()
    res = torch.cuda.FloatTensor(len_b,len_m, 1).fill_(0)
    isParallel = isParallelMatrix(a0, a1, b0, b1, 2)
    # ic(isParallel.shape)
    # ic(res.shape)
    # ic(a0.shape)
    # ic(a1.shape)
    # ic(b0.shape)
    # ic(b1.shape)
    

    res[~isParallel] = shortest_dist_non_parallel_matrix(a0, a1, b0, b1, 2).view([len_b, len_m, 1])[~isParallel]
    res[isParallel] = shortest_dist_parallel_matrix(a0, a1, b0, b1, 2).view([len_b, len_m, 1])[isParallel]
    res = res.view(len_b, n_t, n_t)
    return res

def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()

# TODO: check that this actually does what we want it to do, and take in info from other batches
class FeatureBatchNorm(nn.Module):
    def __init__(self):
        super(FeatureBatchNorm, self).__init__()
        self._bn = nn.BatchNorm2d(1, momentum=0.01, eps=0.001)

    def forward(self, X):
        return self._bn(X.unsqueeze(1)).squeeze(1)


class ParticleNetLaplace(nn.Module):
    def __init__(
            self,
            conv_params: Tuple[EdgeConvSpec],
            mlp_params: Tuple[PredMLPSpec],
            final_pooling: str,
            input_dim: int,
            hidden_dim, 
            hidden_activation='Tanh', 
            layer_norm=True,
            affinity_loss=False, 
            affinity_loss_CE_weight=0.1, 
            affinity_loss_Lp_weight=0.1,
            affinity_loss_frobenius_weight=0.1,
            affinity_loss_11_weight=0.1,
            d_metric='einsum',
            GNN_config={},
            **kwargs
            ):
        """
        conv_params: A tuple where each element is of the form (K, mlp_dims,
        activation). K is the number of neighbors used in that edge convolution
        layer. mlp_dims is a tuple where each element of the tuple specifies the
        dimension of the MLP layer. 'activation' is a string which specifies the

        activation of that MLP layer in the edge convolution layer.
        mlp_params: A tuple where each element is of the form (S, dropout_rate).
        S is the size of that layer in the MLP. dropout_rate is the dropout rate
        for that layer in the MLP.

        final_pooling: A string that is either 'mean', 'max', 'gat', or 'diffpool'.
        Represents how the final pooling is done at the end of several
        applications of the edge convolution.

        input_dim: An integer which specifies the number of features in the
        input.
        """
        super(ParticleNetLaplace, self).__init__()
        self._conv_params = conv_params
        self._mlp_params = mlp_params
        self._final_pooling = final_pooling

        self._ft_bn = FeatureBatchNorm()
        edge_convs = []
        prev_dim = input_dim
        for conv_param in conv_params:
            K, point_indices, mlp_dims, conv_pooling, activation = conv_param
            edge_convs.append(
                    EdgeConvolution(
                        K,
                        point_indices,
                        prev_dim,
                        mlp_dims,
                        conv_pooling,
                        activation,
                        d_metric
                    )
            )
            prev_dim = mlp_dims[-1]
        self._edge_convs = nn.Sequential(*edge_convs)

        self._pooling = ParticleNetPooling(final_pooling)
        final_mlp_layers = []
        final_mlp_layers_linkage = []
        for mlp_param in mlp_params:
            dim, drop_rate = mlp_param
            final_mlp_layers.append(nn.Linear(prev_dim, dim))
            # Change: added batchnorm
#            final_mlp_layers.append(nn.BatchNorm1d(dim))
            final_mlp_layers.append(nn.ReLU())
            final_mlp_layers.append(nn.Dropout(drop_rate))
            prev_dim = dim

            self._prev_dim = prev_dim
        final_mlp_layers.append(nn.Linear(prev_dim, prev_dim))

        self._final_mlp = nn.Sequential(*final_mlp_layers)
        self._layers = nn.Sequential(
                self._ft_bn,
                *edge_convs,
                self._pooling,
                *final_mlp_layers
        )

        self._layers_e = nn.Sequential(
                self._ft_bn,
                *edge_convs,
        )
        self._layers_p = nn.Sequential(
                self._pooling,
                *final_mlp_layers
        )

        
        self._single_mlp = nn.Linear(input_dim, 2)

        # loss function
        self.loss_func = nn.BCELoss()
        # self.loss_func = binary_cross_entropy
        self.sigmoid = nn.Sigmoid()
        # affinity loss
        self.affinity_loss = affinity_loss
        self.affinity_loss_CE_weight = affinity_loss_CE_weight
        self.affinity_loss_Lp_weight = affinity_loss_Lp_weight
        self.affinity_loss_frobenius_weight = affinity_loss_frobenius_weight
        self.affinity_loss_11_weight = affinity_loss_11_weight

        # Pairwise Predictor
        self.pairwise_predictor = Pairwise_Predictor(GNN_config['hidden_dim'], hidden_dim, hidden_activation, layer_norm)

        logging.info(
        'Model: \n%s\nParameters: %i' %
        (self.pairwise_predictor, sum(p.numel() for p in self.pairwise_predictor.parameters())))

    def forward(self, X):
        
        # if X.shape[-1] > 1:
        #     pred = self._layers(X)
        # else:
        #     pred = self._single_mlp(X.squeeze(-1).tranpose(0, -1))
        # ic(pred.shape)

        # X: BxCxN
        pred_x = self._layers_e(X)
        # ic(pred.shape)

        pred_A = self.pairwise_predictor(pred_x)

        # pred_x = self._layers_p(pred_x)

        # dim_out = X.shape[-1]
        # layer_linear = nn.Linear(pred.shape[-1], dim_out).to('cuda')
        # pred = layer_linear(pred)
        # ic(pred.shape)

        # pred = pred.transpose(1, 2)

        return pred_x, pred_A

    def get_loss(self, A_pred, A_true, track_vtx):
        # loss = self.loss_func(self.sigmoid(y_pred), y_true)
        loss = 0

        A_pred = self.sigmoid(A_pred)

        # ic('get_loss_in_pn')
        # ic(A_pred, A_pred.shape, A_pred.dtype)
        # ic(A_true, A_true.shape, A_true.dtype)

        if self.affinity_loss:
            losses_ce = torch.zeros(A_true.shape[0]).cuda()
            for i_A_true in range(A_true.shape[0]):
                loss_ce_iter = 0
                A_true_iter = A_true[i_A_true]
                is_zero = (A_true_iter == 0)
                is_nonzero = (A_true_iter != 0)
                mask = torch.cuda.FloatTensor(A_pred[0].shape).fill_(1)

                if not (torch.sum(is_zero) == 0 or torch.sum(is_nonzero) == 0 ):
                    if (torch.sum(is_zero) < torch.sum(is_nonzero)):
                        i, j=torch.nonzero(is_nonzero, as_tuple=True)
                        samples=range(i.shape[-1])
                        samples = torch.tensor(sample(samples, torch.sum(is_nonzero) - torch.sum(is_zero)))
                        # i_A_true = torch.tensor([i_A_true]).long()
                        mask[i[samples], j[samples]] = 0
                    elif (torch.sum(is_zero) > torch.sum(is_nonzero)):
                        i, j=torch.nonzero(is_zero, as_tuple=True)
                        samples=range(i.shape[-1])
                        samples = torch.tensor(sample(samples, torch.sum(is_zero) - torch.sum(is_nonzero)))
                        # i_A_true = torch.tensor([i_A_true]).long()
                        # ic(i_A_true.dtype)
                        mask[i[samples], j[samples]] = 0
                    mask = mask.to(torch.bool)
                    # masked_A_pred = A_pred[mask]
                    # masked_A_true = A_true[mask]
                    # ic(A_pred.shape)
                    # ic(i_A_true)
                    # ic(mask.shape)
                    loss_ce_iter = self.affinity_loss_CE_weight*self.loss_func(A_pred[i_A_true][mask], A_true[i_A_true][mask]) 
    
                else:
                # if (torch.sum(is_zero) == 0 or torch.sum(is_nonzero) == 0 ):
                    loss_ce_iter = self.affinity_loss_CE_weight*self.loss_func(A_pred[i_A_true], A_true[i_A_true]) 
                if loss_ce_iter.isnan():
                    ic(A_pred.shape)
                    ic(torch.sum(mask))
                    ic(torch.sum(A_true == 0))
                    loss_ce_iter = torch.zeros(1).cuda()
                losses_ce[i_A_true] = loss_ce_iter
            loss_ce = torch.sum(losses_ce)
            # ic(self.affinity_loss_CE_weight, self.affinity_loss_Lp_weight)
            # track_vtx.shape: N_batch, N_track, 3
            loss_lp = self.affinity_loss_Lp_weight*self.get_normalized_laplacian_loss(track_vtx, A_pred)
            # ic(loss_ce, loss_lp)
            # loss_ce = torch.square(loss_ce)
            loss_frobenius = self.affinity_loss_frobenius_weight*torch.sum(torch.sqrt(torch.sum(A_pred**2, dim=(-1, -2))))
            loss_11 = self.affinity_loss_11_weight*torch.sum(torch.abs(A_pred))
            loss = loss_ce + loss_lp + loss_frobenius + loss_11
        return loss, loss_ce, loss_lp, loss_frobenius, loss_11

    def get_normalized_laplacian_loss(self, O, A):
        v=torch.sum(A,dim=1)
        D=torch.diag_embed(v)
        Glap=D-A

        # ic(O, O.shape, O.dtype)
        # ic(A, A.shape, A.dtype)
        # ic(D, D.shape, D.dtype)
        # ic(Glap, Glap.shape, Glap.dtype)

        # normalized_Glap = torch.bmm(torch.bmm(torch.sqrt(D), Glap), torch.sqrt(D))
        # ic(normalized_Glap.shape)

        ans= torch.bmm(O.transpose(2,1), Glap.double())

        ans=torch.bmm(ans, O) # O^T * Graph laplacian * O
        loss=torch.sum(torch.diagonal(ans,dim1=1,dim2=2))
        return loss

class EdgeConvolution(nn.Module):
    def __init__(self,
            K: int,
            point_indices: PointSpec,
            input_dim: int,
            mlp_dims: EdgeMLPSpec,
            conv_pooling: str,
            activation: str,
            d_metric: str
    ):
        super(EdgeConvolution, self).__init__()
        self._K_init = K
        self._k = K
        self._point_indices = torch.LongTensor(point_indices).cuda()
        self._input_dim = input_dim
        self._mlp_dims = mlp_dims
        self._activation = activation
        self._conv_pooling = conv_pooling
        self._d_metric = d_metric

        mlp_layers = []
        prev_dim = input_dim * 2
        activation = getattr(nn.modules.activation, activation)
        self._final_activation = activation()
        # MLP expects features to be in the first dimension
        for dim in mlp_dims:
            mlp_layers.append(nn.Conv2d(prev_dim, dim, 1))
            mlp_layers.append(activation())
            mlp_layers.append(nn.BatchNorm2d(dim, momentum=0.01, eps=0.001))
            prev_dim = dim

        self._mlp = nn.Sequential(*mlp_layers)
        self._pooling = EdgeConvolutionPooling(conv_pooling)
        self._sc = nn.Conv1d(input_dim, mlp_dims[-1], 1, bias=False)
        self._sc_bn = nn.BatchNorm1d(mlp_dims[-1], momentum=0.01, eps=0.001)

    def forward(self, X):
        """
        X: Tensor of shape [n_minibatches, n_track_features, n_tracks]
        """
#        X = X.transpose(-1, -2)
        points = X.index_select(1, self._point_indices)
        deltas = points.transpose(-1, 0).unsqueeze(1) - points.transpose(-1, 0)
        self._points = points
        self._deltas = deltas
        # Need to calculate distances
        # Need to calculate distance between points

        # shape: (n_minibatches, n_tracks, n_tracks)
        K = min(X.shape[-1] - 1, self._K_init)
        self._K = K

        # point_indices, 3 3d points -> 9
        # points: [n_minibatches, point_indices, n_tracks]
        # deltas: [n_tracks, n_tracks, point_indices, n_minibatches]
        # distances: [n_minibatches, n_tracks, n_tracks]

        if self._d_metric == 'intertrack':
            distances = get_distance(points)
            for i in range(len(distances)):
                distances[i].fill_diagonal_(torch.max(distances).item() + 1)
            
            nearest_neighbors = torch.topk(distances, K, dim=-1, sorted=True, largest=False)[1][:,:,:]
        
        elif self._d_metric == 'einsum':
        # j: references track index
        # k: target track index
        # m: minibatch index
        # f: feature index
        
            distances = torch.einsum("jkfm,jkfm->mjk", deltas, deltas)

            self._distances = distances
            # Shape: (n_minibatches, n_tracks, K)
            nearest_neighbors = torch.topk(distances, K + 1, dim=-1, sorted=True, largest=False)[1][:,:,1:]

        # ic(distances.shape)
        # ic(K)
        # ic(nearest_neighbors.shape)

        self._nearest_neighbors = nearest_neighbors
        # Now we have the nearest vertex for each vertex
        # we have the indices of the closets neighbors (n_minibatches, n_tracks, K)
        # X_nn: (n_minibatches, n_tracks, K, n_features)
        X_center = X.unsqueeze(-2).repeat(1, 1, K, 1)
        self._X_center = X_center
        X_nn = X_center.gather(-1, nearest_neighbors.unsqueeze(1).repeat(1, X.shape[1], 1, 1).transpose(-1, -2))
        self._X_nn = X_nn
        H = torch.cat((X_nn - X_center, X_center), dim=1)
        self._H_1 = H
        H = self._pooling(self._mlp(H)) + self._sc_bn(self._sc(X))
        self._H_2 = H
        return self._final_activation(H)

class EdgeConvolutionPooling(nn.Module):
    def __init__(
            self, 
            conv_pooling: str
        ):
        super(EdgeConvolutionPooling, self).__init__()
        self._conv_pooling = conv_pooling

    def forward(self, X):
        """
        X: Tensor of shape [n_minibatches, n_features, K, n_tracks]
        """
        if self._conv_pooling == 'max':
            return torch.max(X, dim=2)
        elif self._conv_pooling == 'mean':
            return torch.mean(X, dim=2)
        else:
            raise NotImplementedError(f'Have not implemented {self._conv_pooling} pooling')

class ParticleNetPooling(nn.Module):
    def __init__(
            self, 
            conv_pooling: str
        ):
        super(ParticleNetPooling, self).__init__()
        self._conv_pooling = conv_pooling

    def forward(self, X):
        """
        X: Tensor of shape [n_minibatches, n_features, K, n_tracks]
        """
        if self._conv_pooling == 'max':
            return torch.max(X, dim=-1)
        elif self._conv_pooling == 'mean':
            return torch.mean(X, dim=-1)
        else:
            raise NotImplementedError(f'Have not implemented {self._conv_pooling} pooling')

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
        # ic(x.shape)
        # x = x.transpose(2, 1)  # from BxNxC to BxCxN

        n = x.shape[2]
        # ic(n)
        # ic(x.shape)
        # broadcast
        m1 = x.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to rows  
        m2 = x.unsqueeze(3).repeat(1, 1, 1, n)  # broadcast to cols  
        block = torch.cat((m1, m2), dim=1) 

        # print(f'block: {block}')
        # ic(block.shape)
        # predict affinity and sv
        edge_vals = self.mlp_affinity(block) # shape (B,out_features,N,N)
        # cds=self.mlp_coord(x.transpose(2, 1))
        edge_vals = edge_vals.squeeze(1)
        edge_vals = edge_vals + edge_vals.transpose(1, 2)

        # ic(edge_vals.shape)
        return edge_vals
