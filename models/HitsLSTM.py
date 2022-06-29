import torch.nn as nn
import torch

class HitsLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, hidden_size, n_classes, hidden_activation):
        super(HitsLSTM, self).__init__()
        self.rnn = nn.LSTM(n_features, hidden_size, n_hidden,  bidirectional=True)

        # self.args = args
        # self._num_features = num_features
        # self._nhid = nhid
        # self._num_classes = num_classes
        # self._pooling_ratio = pooling_ratio
        act = getattr(nn, hidden_activation)
        layers = [nn.Linear(hidden_size*8, hidden_size), act()]
        for i in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act())

        layers.append(nn.Linear(hidden_size, n_classes))

        self.transform = nn.Sequential(*layers)

    def forward(self, x):
        hits = x[:, :, :15]
        hits = hits.reshape(hits.shape[0]*hits.shape[1], 5, 3).transpose(0, 1)
        emb = x[:, :, 15:]
        emb = emb.reshape(x.shape[0]*emb.shape[1], 1, emb.shape[-1]).repeat(1, 5, 1).transpose(0, 1)
        inp = torch.cat((hits, emb), dim=-1)
        output, (hn, cn) = self.rnn(inp)
        res = torch.cat((output[-1], hn[-1], cn[-1], output[0], hn[0], cn[0]), dim=-1)
        res = res.reshape(x.shape[0], x.shape[1], res.shape[-1])

        return self.transform(res)
