from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature
import torch

class TrackHitDropping(Augmentor):
    def __init__(self, pf: float):
        super(TrackHitDropping, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_track_hits(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

def drop_track_hits(x: torch.Tensor, drop_prob) -> torch.Tensor:
    device = x.device
    x = x.clone()
    for i, p in enumerate(drop_prob):
        drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < p
        drop_mask = drop_mask.to(device)
        x[drop_mask, (i+1)*3:15] = 0

    return x
