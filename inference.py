import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn

import sklearn.metrics as metrics
from scipy.sparse import coo_matrix
from scipy.stats import mode


import matplotlib.cm as cm
from utils.log import load_config
from utils.log import load_checkpoint

DEVICE = 'cuda'

# Set Transformer Inference

config_file = '/home1/tingtingxuan/sunrise-trigger/train_results/set/experiment_2022-03-03_14:28:24/config.pkl'
config = pickle.load(open(config_file, 'rb'))

from models.SetTransformer import SetTransformer

config['model']['dim_input'] += config['data']['use_radius']
model = SetTransformer(**config['model'])

model = model.to(DEVICE)

result_dir = '/home1/tingtingxuan/sunrise-trigger/train_results/set/experiment_2022-03-03_14:28:24'
checkpoint_dir = os.path.join(result_dir, 'checkpoints')
checkpoint_file = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint')])
checkpoint_file = checkpoint_file[-1]
print(checkpoint_file)
model = load_checkpoint(checkpoint_file, model)
print('Successfully reloaded!')

# Test Settings
#test_dir1 = os.path.expandvars('physics_data/nontrigger_event/NN')
test_dir1 = '/home1/tingtingxuan/Data/predicted_track_7layer_with_momentum_large/non-trigger'
test_dir2 = '/home1/tingtingxuan/Data/predicted_track_7layer_with_momentum_large/trigger'
#test_dir1 = os.path.expandvars('physics_data/trigger_event')
test_samples1 = 3
test_samples2 = 3
# test_samples1 = 10000
# test_samples2 = 10000
batch_size = 32

# Load testing data
from dataloaders.predicted_trkvec import TrkDataset
from dataloaders.predicted_trkvec import JetsBatchSampler
from torch.utils.data import DataLoader

test_dataset = TrkDataset(input_dir=test_dir1, n_samples=test_samples1, n_input_dir=2, input_dir2=test_dir2, n_samples2=test_samples2)
test_batch_sampler = JetsBatchSampler(test_dataset.n_tracks, batch_size)
test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)
print('Loaded %g inference samples' % len(test_data_loader.dataset))

# Inference part
model.eval()
test_loss = 0

preds = []
labels = []
preds_prob = []
count = 0
num_insts = 0
    
for batch in test_data_loader:
    tracks, vtx, partitions_as_graph, n_tracks, trig, energy, momentum, is_trigger_track, radii = batch
    count += 1
    tracks = tracks.to(DEVICE, torch.float)
    trig = (trig.to(DEVICE) == 1).long()
    batch_size = tracks.shape[0]
    num_insts += batch_size

    labels.append(trig.long().cpu().numpy())
    
    if config['data']['use_radius']:
        radii = radii.to(DEVICE)
        radii = radii.to(tracks.dtype)
        tracks = torch.cat((tracks, radii), dim=-1)


    with torch.set_grad_enabled(False):
        pred_labels = model(tracks).view(-1, 2)
        pred = pred_labels.max(dim=1)[1]
        preds.append((pred).cpu().data.numpy())
        preds_prob.extend(nn.Softmax(dim=1)(pred_labels)[:, 1].detach().cpu().numpy().flatten())
        loss = F.nll_loss(pred_labels, trig)
    test_loss += loss.item() * batch_size


labels = np.hstack(labels)
preds = np.hstack(preds)
preds_prob = np.hstack(preds_prob)

result = {'prec': metrics.precision_score(labels, preds>0),
            'recall': metrics.recall_score(labels, preds>0),
            'acc': metrics.accuracy_score(labels, preds>0),
            'F1': metrics.f1_score(labels, preds>0),
            'auroc': metrics.roc_auc_score(labels, preds_prob)}

print(result)

print(f'Trigger: {sum(labels==1)} Non-Trigger: {sum(labels==0)}')

def check_efficiency_and_purity(drop_rate, preds_prob, labels):
    threshold = np.quantile(preds_prob, drop_rate)
    predictions = (preds_prob > threshold)
    cm = metrics.confusion_matrix(predictions, labels)
    tp = cm[1][1]
    tn = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    # print(cm)
    efficiency = tp / (tp + fn)
    purity = tp / (tp + fp)
    print(f'Input {np.round(100*test_samples2/(test_samples1+test_samples2))}% Trigger Events \t drop_rate: {np.round(100*drop_rate,2)}% \t efficiency: {np.round(100*efficiency, 2)}% \t purity: {np.round(100*purity, 2)}%')

check_efficiency_and_purity(0.9, preds_prob, labels)
check_efficiency_and_purity(0.95, preds_prob, labels)
check_efficiency_and_purity(0.99, preds_prob, labels)
check_efficiency_and_purity(1-1/150, preds_prob, labels)