"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'hitgraphs_sparse_new':
        from torch_geometric.data import Batch
        from . import hitgraphs_sparse_new
        train_dataset, valid_dataset = hitgraphs_sparse_new.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    elif name == 'trkset':
        from torch_geometric.data import Batch
        from . import trkvec
        train_dataset, valid_dataset = trkvec.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    elif name == 'trkvec':
        from . import trkvec_sampler
        return trkvec_sampler.get_data_loaders(name, batch_size, **data_args)
    elif name == 'trkvec-ecml':
        from . import trkvec_ecml
        return trkvec_ecml.get_data_loaders(name, batch_size, **data_args)
    elif name == 'trkvec_predicted':
        from . import predicted_trkvec
        return predicted_trkvec.get_data_loaders(name, batch_size, **data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
        valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                   shuffle=(train_sampler is None), **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader
