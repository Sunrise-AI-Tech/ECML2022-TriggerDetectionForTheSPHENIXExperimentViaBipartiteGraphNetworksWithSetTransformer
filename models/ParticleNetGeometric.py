import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Tuple
import torch_scatter
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import reset
from typing import Tuple

ActivationSpec = str
PoolingSpec = str
EdgeMLPSpec = Tuple[int]
PointSpec = Tuple[int]
EdgeConvSpec = Tuple[int, PointSpec, EdgeMLPSpec, PoolingSpec, ActivationSpec]
PredMLPSpec = Tuple[int, float]

def make_conv_mlp(input_dim, mlp_dims, activation):
    layers = []
    prev_dim = input_dim
    for mlp_dim in mlp_dims:
        layers.append(nn.Linear(prev_dim, mlp_dim))
        layers.append(getattr(nn, activation)())
        prev_dim = mlp_dim

    return nn.Sequential(*layers)

class ParticleNetGeometric(nn.Module):
    def __init__(
            self,
            conv_params: Tuple[EdgeConvSpec],
            mlp_params: Tuple[PredMLPSpec],
            final_pooling: str,
            input_dim: int,
            learning_rate: float
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
        super(ParticleNetGeometric, self).__init__()
        self._conv_params = conv_params
        self._mlp_params = mlp_params
        self._final_pooling = final_pooling
        self.name = 'ParticleNetGeometric'
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        def lr_schedule(epoch):
            if epoch > 10 and epoch <= 20:
                return 0.1
            elif epoch > 20 and epoch <= 40:
                return 0.01
            elif epoch > 40 and epoch <= 80:
                return 0.001
            elif epoch > 80:
                return 0.0001
            else:
                return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)


        edge_convs = []
        prev_dim = input_dim
        for conv_param in conv_params:
            K, point_indices, mlp_dims, conv_pooling, activation = conv_param
            conv_mlp = make_conv_mlp(2 * prev_dim, mlp_dims, activation)
            edge_convs.append((
                        FilteredDynamicEdgeConv(
                            conv_mlp,
                            prev_dim,
                            mlp_dims[-1],
                            K,
                            point_indices,
                            conv_pooling,
                            ), 'x, edge_index, batch -> x'
                        )
                    )
            edge_convs.append((
                gnn.InstanceNorm(mlp_dims[-1]), 
                'x, batch -> x'
            ))
            edge_convs.append(torch.nn.ReLU(inplace=True))

            prev_dim = mlp_dims[-1]
        self._edge_convs = gnn.Sequential('x, edge_index, batch -> x', edge_convs)
        self.loss_func = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        self._pooling = ParticleNetPooling(prev_dim, final_pooling)
        prev_dim = self._pooling.get_output_dim()
        final_mlp_layers = []
        for mlp_param in mlp_params:
            dim, drop_rate = mlp_param
            final_mlp_layers.append(nn.Linear(prev_dim, dim))
            # Change: added batchnorm
#            final_mlp_layers.append(nn.BatchNorm1d(dim))
            final_mlp_layers.append(nn.ReLU())
            final_mlp_layers.append(nn.Dropout(drop_rate))
            prev_dim = dim
        final_mlp_layers.append(nn.Linear(prev_dim, 2))

        self._final_mlp = nn.Sequential(*final_mlp_layers)
        self._layers = gnn.Sequential('x, edge_index, batch', [
#                (self._ft_bn, 'x -> x'),
                *edge_convs,
            ]
        )


    def forward(self, x, edge_index, batch):
        x = self._layers(x, edge_index, batch)
        x = self._pooling(x, batch)
        x = self._final_mlp(x)
        return x[:, 1] - x[:, 0] # in order to simulate single-class MLP

    def train_model(self, trig_pred, trig_true):
        self.optimizer.zero_grad()
        loss = self.get_loss(trig_pred, trig_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, trig_pred, trig_true):
        return self.loss_func(self.sigmoid(trig_pred), ip_true)



class ParticleNetPooling(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            pooling: str
        ):
        super(ParticleNetPooling, self).__init__()
        self._pooling = pooling
        self._input_dim = input_dim

    def get_output_dim(self):
        if self._pooling in {'mean', 'max'}:
            return self._input_dim
        elif self._pooling in 'meanmax':
            return 2 * self._input_dim
        else:
            raise NotImplementedError(f'Have not implemented {self._pooling} pooling')

    def forward(self, x, batch):
        if self._pooling == 'max':
            pooled = torch_scatter.scatter_max(x, batch, dim=0)
        elif self._pooling == 'mean':
            pooled = torch_scatter.scatter_mean(x, batch, dim=0)
        elif self._pooling == 'meanmax':
            pooled_mean = torch_scatter.scatter_mean(x, batch, dim=0)
            pooled_max = torch_scatter.scatter_max(x, batch, dim=0)[0]
            pooled = torch.cat((pooled_mean, pooled_max), dim=-1)
        else:
            raise NotImplementedError(f'Have not implemented {self._pooling} pooling')

        return pooled

try:
    from torch_cluster import knn
except ImportError:
    knn = None

class FilteredDynamicEdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
                \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
            self, nn: Callable, 
            input_dim,
            output_dim,
            k: int, 
            point_indices: PointSpec,
            aggr: str = 'max', 
            num_workers = 1,
            enable_knn = False,
            **kwargs
            ):
        super(FilteredDynamicEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.sc = torch.nn.Linear(input_dim, output_dim)
        self.point_indices = torch.Tensor(point_indices).long().cuda()
        self.enable_kinn = enable_knn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)


    def forward(self, 
            x: Union[Tensor, PairTensor], 
            edge_index: Adj,
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""
        n = x.shape[0]
        assert n <= 3037000499, 'Maximum value supported by encoding scheme exceeded'
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        assert x[0].dim() == 2, \
                'Static graphs not supported in `DynamicEdgeConv`.'


        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        if self.enable_knn:
            p = x[0].index_select(-1, self.point_indices)
            knn_edge_index = knn(p, p, self.k, b[0], b[1],
                    num_workers=self.num_workers)
            knn_edge_index = knn_edge_index[0] * n + knn_edge_index[1]
            edge_index = edge_index[0] * n + edge_index[1]
            combined = torch.cat((knn_edge_index, edge_index))
            uniques, counts = combined.unique(return_counts=True)
            edge_index = torch.stack((uniques[counts > 1] // n, uniques[counts > 1] % n), dim=0)

        xp = self.propagate(edge_index, x=x, size=None)
        return xp + self.sc(x[0])


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
