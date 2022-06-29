import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features, n_hidden, hidden_size, n_classes, hidden_activation):
        super(MLP, self).__init__()

        # self.args = args
        # self._num_features = num_features
        # self._nhid = nhid
        # self._num_classes = num_classes
        # self._pooling_ratio = pooling_ratio
        act = getattr(nn, hidden_activation)
        layers = [nn.Linear(n_features, hidden_size), act()]
        for i in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act())

        layers.append(nn.Linear(hidden_size, n_classes))

        self.transform = nn.Sequential(*layers)

    def forward(self, x):
        return self.transform(x)
