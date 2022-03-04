from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature
import torch
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.utils import dropout_adj, subgraph


class BackgroundTrackDropping(Augmentor):
    def __init__(self, pf: float):
        super(BackgroundTrackDropping, self).__init__()
        self.pf = pf
        self.trigger_tracks = None

    def augment(self, g: Graph) -> Graph:
        if self.trigger_tracks is None:
            return g
            
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = drop_background_tracks(edge_index, self.trigger_tracks, edge_weights, keep_prob=1. - self.pf, x=x)
        self.trigger_tracks = None

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

    def set_trigger_tracks(self, trigger_tracks):
        self.trigger_tracks = trigger_tracks

def drop_background_tracks(edge_index: torch.Tensor, trigger_tracks, edge_weight=None, keep_prob: float = 0.5, x=None):
    num_nodes = trigger_tracks.shape[0]
    probs = torch.tensor([keep_prob for _ in range(num_nodes)])
    dist = Bernoulli(probs)

    keep = (dist.sample().to(trigger_tracks.device) + trigger_tracks) >= 1
    subset = keep.to(torch.bool).to(edge_index.device)
    edge_index, edge_weight = subgraph(subset, edge_index, edge_weight)

    return edge_index, edge_weight
