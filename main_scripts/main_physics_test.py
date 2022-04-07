import random
import os
import sys
import argparse
import copy
import shutil
import json
import logging
import yaml
import pickle
from pprint import pprint
from datetime import datetime
from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from icecream import ic
from collections import defaultdict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch_geometric

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from scipy.stats import pearsonr, spearmanr
from models.MLP import MLP
from models.HitsLSTM import HitsLSTM
# from models.ParticleNetLaplaceDiffpool import ParticleNetLaplaceDiffpool
from models.ParticleNetLaplace import ParticleNetLaplace
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score

DEVICE = 'cuda:1'
OLD_COLUMNS = None

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/physics.yaml')
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('--auto', action='store_true')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--show-config', action='store_true')
    argparser.add_argument('--resume', action='store_true', default=0, help='Resume from last checkpoint')
    
    # Logging
    argparser.add_argument('--name', type=str, default=None,
            help="Run name")
    argparser.add_argument('--wandb', type=str, default='trigger-pns', 
            help="wandb project name")
    argparser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    argparser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

    args = argparser.parse_args()

    return args

def extract_hyperparameters(config):
    hp = {}
    hp['type/optimizer'] = config['optimizer']['type']
    hp['momentum/optimizer'] = config['optimizer']['momentum']
    hp['weight_decay/optimizer'] = config['optimizer']['weight_decay']
    hp['learning_rate/optimizer'] = config['optimizer']['learning_rate']
    hp['name/data'] = config['data']['name']
    hp['n_train/data'] = config['data']['n_train']
    hp['n_valid/data'] = config['data']['n_valid']
    hp['n_test/data'] = config['data']['n_valid']
    hp['batch_size/data'] = config['data']['batch_size']
    hp['load_complete_graph/data'] = config['data']['load_complete_graph']
    hp['add_geo_features/data'] = config['data']['add_geo_features']
    hp['use_momentum/data'] = config['data']['use_momentum']
    hp['use_energy/data'] = config['data']['use_energy']
    hp['use_pt/data'] = config['data']['use_pt']
    hp['use_radius/data'] = config['data']['use_radius']

    hp['type/model'] = config['model']['type']
    hp['n_features/model'] = config['model']['n_features']
    hp['n_hidden/model'] = config['model']['n_hidden']
    hp['hidden_size/model'] = config['model']['hidden_size']
    hp['hidden_activation/model'] = config['model']['hidden_activation']


    return hp

def train(data, model, optimizer, epoch, output_dir, loss='mse', use_energy=False, use_momentum=False, use_pt=True, use_radius=False):
    train_info = do_epoch(data, model, epoch, optimizer, loss=loss, use_energy=use_energy, use_momentum=use_momentum, use_pt=use_pt, use_radius=use_radius) 
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model, epoch, loss='mse', use_energy=False, use_momentum=False, use_pt=True, use_radius=False):
    with torch.no_grad():
        val_info = do_epoch(data, model, epoch, optimizer=None, use_energy=use_energy, use_momentum=use_momentum, use_pt=use_pt, use_radius=use_radius) 
    return val_info


def do_epoch(data, model, epoch, optimizer=None, loss='mse', use_energy=False, use_momentum=False, use_pt=True, use_radius=False):
    if optimizer is None:
        # validation epoch
        model.eval()
        phase = 'validation'
    else:
        # train epoch
        model.train()
        phase = 'train'

    if loss == 'mse':
        loss_fn = nn.MSELoss()
    elif loss == 'mae':
        loss_fn = nn.MAELoss()
    else:
        raise NotImplementedError(f'Loss function of type {loss} not implemented.')

    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in (
        'loss',
        'mse',
        'mae',
        'p_t_nz_mae',
        'p_t_nz_mse',
        'p_t_nz_r2',
        'p_t_nz_pearson_r',
        'p_t_nz_pearson_p',
        'p_t_nz_spearman_r',
        'p_t_mae',
        'p_t_mse',
        'p_t_r2',
        'p_t_spearman_r',
        'p_t_spearman_p',
    )}



    num_insts = 0
    skipped_batches = 0
    preds = []
    correct = []
    
    for batch in data:
        tracks, _, _, _, _, energy, momentum, _, radii = batch
        tracks = tracks.to(DEVICE, torch.float)
        radii = radii.to(DEVICE, torch.float)
        batch_size = tracks.shape[0]        
        energy = energy.to(DEVICE)
        momentum = momentum.to(DEVICE)

        num_insts += batch_size
        
        if use_radius:
            tracks = torch.cat((tracks, radii), dim=-1)
        
        physics_pred = model(tracks)
        pp = physics_pred.detach().cpu().numpy()
        pp = np.reshape(pp, (-1, pp.shape[-1]))
        preds.extend(pp)

        energy = energy.to(tracks.dtype)
        momentum = momentum.to(tracks.dtype)
        pts = torch.sqrt(torch.sum(momentum[..., :2]**2, dim=-1)).unsqueeze(-1)

        gts = []
        if use_energy:
            gts.append(energy)
        if use_momentum:
            gts.append(momentum)
        if use_pt:
            gts.append(pts)
            
        physics_true = torch.cat(gts, dim=-1)
        pt = physics_true.detach().cpu().numpy()
        pt = np.reshape(pt, (-1, pt.shape[-1]))
        correct.extend(pt)

        index = 0
        losses = []
        if use_energy:
            pp = physics_pred[..., index:index+3]
            pt = physics_true[..., index:index+3]
            accum_info['e_mse'] += torch.sum((pp[..., 0:3] - pt[..., 0:3])**2).item()
            accum_info['e_mae'] += torch.sum(torch.abs(pp[..., 0:3] - pt[..., 0:3])).item()
            accum_info['e_x_mse'] += torch.sum((pp[..., 0] - pt[..., 0])**2).item()
            accum_info['e_x_mae'] += torch.sum(torch.abs(pp[..., 0] - pt[..., 0])).item()
            accum_info['e_y_mse'] += torch.sum((pp[..., 1] - pt[..., 1])**2).item()
            accum_info['e_y_mae'] += torch.sum(torch.abs(pp[..., 1] - pt[..., 1])).item()
            accum_info['e_z_mse'] += torch.sum((pp[..., 2] - pt[..., 2])**2).item()
            accum_info['e_z_mae'] += torch.sum(torch.abs(pp[..., 2] - pt[..., 2])).item()
            index += 3
        if use_momentum:
            pp = physics_pred[..., index:index+3]
            pt = physics_true[..., index:index+3]
            accum_info['p_mse'] += torch.sum((pp[..., 0:3] - pt[..., 0:3])**2).item()
            accum_info['p_mae'] += torch.sum(torch.abs(pp[..., 0:3] - pt[..., 0:3])).item()
            accum_info['p_x_mse'] += torch.sum((pp[..., 0] - pt[..., 0])**2).item()
            accum_info['p_x_mae'] += torch.sum(torch.abs(pp[..., 0] - pt[..., 0])).item()
            accum_info['p_y_mse'] += torch.sum((pp[..., 1] - pt[..., 1])**2).item()
            accum_info['p_y_mae'] += torch.sum(torch.abs(pp[..., 1] - pt[..., 1])).item()
            accum_info['p_z_mse'] += torch.sum((pp[..., 2] - pt[..., 2])**2).item()
            accum_info['p_z_mae'] += torch.sum(torch.abs(pp[..., 2] - pt[..., 2])).item()
            index += 3
        if use_pt:
            pp = physics_pred[..., index:index+3]
            pt = physics_true[..., index:index+3]
            accum_info['p_t_mse'] += torch.sum((pp[..., 0] - pt[..., 0])**2).item()
            accum_info['p_t_mae'] += torch.sum(torch.abs(pp[..., 0] - pt[..., 0])).item()
            accum_info['p_t_nz_mse'] += torch.sum((pp[..., 0] != 0) * (pp[..., 0] - pt[..., 0])**2).item()
            accum_info['p_t_nz_mae'] += torch.sum((pp[..., 0] != 0) * torch.abs(pp[..., 0] - pt[..., 0])).item()


        loss = loss_fn(physics_pred, physics_true)
        accum_info['loss'] += loss.item() * batch_size
        accum_info['mse'] += torch.sum((physics_pred - physics_true)**2).item()
        accum_info['mae'] += torch.sum((physics_pred - physics_true)).item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info['loss'] += loss.item() * batch_size

    correct = np.array(correct)
    preds = np.array(preds)
    accum_info['loss'] /= num_insts
    accum_info['mse'] /= num_insts
    accum_info['mae'] /= num_insts
    accum_info['p_t_mae'] /= num_insts
    accum_info['p_t_mse'] /= num_insts
    accum_info['p_t_nz_mae'] /= num_insts
    accum_info['p_t_nz_mse'] /= num_insts

    r2 = r2_score(correct, preds, multioutput='raw_values')
    r2_all = r2_score(correct, preds, multioutput='variance_weighted')
    r2_energy = r2_score(correct[:, 0:3], preds[:, 0:3], multioutput='variance_weighted')
    accum_info['r2'] = r2_all
    index = 0
    if use_energy:
        et = correct[..., index:index+3]
        ep = preds[..., index:index+3]
        accum_info['e_r2'] = r2_score(et, ep, multioutput='variance_weighted')
        accum_info['e_x_r2'] = r2[index]
        accum_info['e_y_r2'] = r2[index+1]
        accum_info['e_z_r2'] = r2[index+2]
        index += 3
    if use_momentum:
        pt = correct[..., index:index+3]
        pp = preds[..., index:index+3]
        accum_info['p_r2'] = r2_score(pt, pp, multioutput='variance_weighted')
        accum_info['p_x_r2'] = r2[index]
        accum_info['p_y_r2'] = r2[index+1]
        accum_info['p_z_r2'] = r2[index+2]
        index += 3
    if use_pt:
        accum_info['p_t_r2'] = r2[index]
        r, p = pearsonr(correct.squeeze(), preds.squeeze())
        accum_info['p_t_pearson_r'] = r
        accum_info['p_t_pearson_p'] = p
        r, p = spearmanr(correct.squeeze(), preds.squeeze())
        accum_info['p_t_spearman_r'] = r
        accum_info['p_t_spearman_p'] = p
        r2 = r2_score(correct[preds.squeeze() != 0], preds[preds.squeeze() != 0], multioutput='raw_values')
        accum_info['p_t_nz_r2'] = r2[0]
        r, p = pearsonr(correct[preds.squeeze() != 0].squeeze(), preds[preds.squeeze() != 0].squeeze())
        accum_info['p_t_nz_pearson_r'] = r
        accum_info['p_t_nz_pearson_p'] = p
        r, p = spearmanr(correct[preds.squeeze() != 0].squeeze(), preds[preds.squeeze() != 0].squeeze())
        accum_info['p_t_nz_spearman_r'] = r
        accum_info['p_t_nz_spearman_p'] = p

        index += 1

    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)

    return accum_info

def main(auto=False, parser_dict=None, trails_number=None, datasets=None):
    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

     # Parse the command line
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    hp = extract_hyperparameters(config)

    if auto:
        config['tensorboard_output_dir'] = parser_dict['tensorboard_output_dir']
    
    config['output_dir'] = os.path.join(config['output_dir'], f'experiment_{start_time:%Y-%m-%d_%H:%M:%S}')
    os.makedirs(config['output_dir'], exist_ok=True)
    config['tensorboard_output_dir'] = os.path.join(config['tensorboard_output_dir'], f'experiment_{start_time:%Y-%m-%d_%H:%M:%S}')

    # Setup logging
    file_handler = config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=0)
    if auto:
        columns = get_terminal_columns()
        logging.info('\n'.join(('',
            "-" * columns,
            f"trail number = {trails_number}",
            "-" * columns
        )))
    logging.info('Command line config: %s' % args)
    logging.info('Configuration: %s', config)
    logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    save_config(config)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.cuda.set_device(int(args.gpu))

    if auto:
        train_data, val_data = datasets
    else:
        # Load data
        logging.info('Loading training data and validation data')
        dconfig = copy.copy(config['data'])

        del dconfig['use_momentum']
        del dconfig['use_pt']
        del dconfig['use_energy']

        train_data, val_data, test_data = get_data_loaders(**dconfig)
        logging.info('Loaded %g training samples', len(train_data.dataset))
        logging.info('Loaded %g validation samples', len(val_data.dataset))
        logging.info('Loaded %g test samples', len(test_data.dataset))


    mconfig = copy.copy(config['model'])
    mconfig['n_classes'] = 3*config['data']['use_energy'] + 3*config['data']['use_momentum'] + config['data']['use_pt']
    del mconfig['type']
    model_paths = (
            'train_results/cached_models/physics_hidden-size-128.pth.tar',
            'train_results/cached_models/physics_hidden-size-256.pth.tar',
            'train_results/cached_models/physics_hidden-size-32.pth.tar',
            'train_results/cached_models/physics_hidden-size-64.pth.tar',
            'train_results/cached_models/physics_hidden-size-8.pth.tar',
    )
    model_sizes = (128, 256, 32, 64, 8)
    for mpath, msize in zip(model_paths, model_sizes):
        mconfig['hidden_size'] = msize
        model = MLP(**mconfig)
        model = model.to(DEVICE)
        model = load_checkpoint(mpath, model)

        print_model_summary(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'The number of model parameters is {num_params}')


        train_info = evaluate(train_data, model, 50,
                loss=config['loss']['type'],
                use_energy=config['data']['use_energy'],
                use_momentum=config['data']['use_momentum'],
                use_pt=config['data']['use_pt'],
                use_radius=config['data']['use_radius'],
                )
        table = make_table(
                ('Loss', f"{train_info['loss']}"),
                ('MSE', f"{train_info['mse']}"),
                ('MAE', f"{train_info['mae']}"),
                ('Tranverse Momentum MAE', f"{train_info['p_t_mae']}"),
                ('Tranverse Momentum MSE', f"{train_info['p_t_mse']}"),
                ('Tranverse Momentum R^2', f"{train_info['p_t_r2']:.6f}"),
                ('Tranverse Momentum Pearson r', f"{train_info['p_t_pearson_r']:.6f}"),
                ('Tranverse Momentum Pearson p', f"{train_info['p_t_pearson_p']:.6f}"),
                ('Tranverse Momentum Spearman r', f"{train_info['p_t_spearman_r']:.6f}"),
                ('Tranverse Momentum Spearman p', f"{train_info['p_t_spearman_p']:.6f}"),
                ('NZ Tranverse Momentum MAE', f"{train_info['p_t_nz_mae']}"),
                ('NZ Tranverse Momentum MSE', f"{train_info['p_t_nz_mse']}"),
                ('NZ Tranverse Momentum R^2', f"{train_info['p_t_nz_r2']:.6f}"),
                ('NZ Tranverse Momentum Pearson r', f"{train_info['p_t_nz_pearson_r']:.6f}"),
                ('NZ Tranverse Momentum Pearson p', f"{train_info['p_t_nz_pearson_p']:.6f}"),
                ('NZ Tranverse Momentum Spearman r', f"{train_info['p_t_nz_spearman_r']:.6f}"),
                ('NZ Tranverse Momentum Spearman p', f"{train_info['p_t_nz_spearman_p']:.6f}"),
                ('Runtime', f"{train_info['run_time']}")
                )

        logging.info('\n'.join((
            '',
            center_text(f"Train hidden_size={msize}", ' '),
            table
        )))
        val_info = evaluate(val_data, model, 50,
                loss=config['loss']['type'],
                use_energy=config['data']['use_energy'],
                use_momentum=config['data']['use_momentum'],
                use_pt=config['data']['use_pt'],
                use_radius=config['data']['use_radius'],
                )
        table = make_table(
                ('Loss', f"{val_info['loss']}"),
                ('MSE', f"{val_info['mse']}"),
                ('MAE', f"{val_info['mae']}"),
                ('Tranverse Momentum MAE', f"{val_info['p_t_mae']}"),
                ('Tranverse Momentum MSE', f"{val_info['p_t_mse']}"),
                ('Tranverse Momentum R^2', f"{val_info['p_t_r2']:.6f}"),
                ('Tranverse Momentum Pearson r', f"{val_info['p_t_pearson_r']:.6f}"),
                ('Tranverse Momentum Pearson p', f"{val_info['p_t_pearson_p']:.6f}"),
                ('Tranverse Momentum Spearman r', f"{val_info['p_t_spearman_r']:.6f}"),
                ('Tranverse Momentum Spearman p', f"{val_info['p_t_spearman_p']:.6f}"),
                ('NZ Tranverse Momentum MAE', f"{val_info['p_t_nz_mae']}"),
                ('NZ Tranverse Momentum MSE', f"{val_info['p_t_nz_mse']}"),
                ('NZ Tranverse Momentum R^2', f"{val_info['p_t_nz_r2']:.6f}"),
                ('NZ Tranverse Momentum Pearson r', f"{val_info['p_t_nz_pearson_r']:.6f}"),
                ('NZ Tranverse Momentum Pearson p', f"{val_info['p_t_nz_pearson_p']:.6f}"),
                ('NZ Tranverse Momentum Spearman r', f"{val_info['p_t_nz_spearman_r']:.6f}"),
                ('NZ Tranverse Momentum Spearman p', f"{val_info['p_t_nz_spearman_p']:.6f}"),
                ('Runtime', f"{val_info['run_time']}")
                )

        logging.info('\n'.join((
            '',
            center_text(f"Validation hidden_size={msize}", ' '),
            table
        )))

        test_info = evaluate(test_data, model, 50,
                loss=config['loss']['type'],
                use_energy=config['data']['use_energy'],
                use_momentum=config['data']['use_momentum'],
                use_pt=config['data']['use_pt'],
                use_radius=config['data']['use_radius'],
                )
        table = make_table(
                ('Loss', f"{test_info['loss']}"),
                ('MSE', f"{test_info['mse']}"),
                ('MAE', f"{test_info['mae']}"),
                ('Tranverse Momentum MAE', f"{test_info['p_t_mae']}"),
                ('Tranverse Momentum MSE', f"{test_info['p_t_mse']}"),
                ('Tranverse Momentum R^2', f"{test_info['p_t_r2']:.6f}"),
                ('Tranverse Momentum Pearson r', f"{test_info['p_t_pearson_r']:.6f}"),
                ('Tranverse Momentum Pearson p', f"{test_info['p_t_pearson_p']:.6f}"),
                ('Tranverse Momentum Spearman r', f"{test_info['p_t_spearman_r']:.6f}"),
                ('Tranverse Momentum Spearman p', f"{test_info['p_t_spearman_p']:.6f}"),
                ('NZ Tranverse Momentum MAE', f"{test_info['p_t_nz_mae']}"),
                ('NZ Tranverse Momentum MSE', f"{test_info['p_t_nz_mse']}"),
                ('NZ Tranverse Momentum R^2', f"{test_info['p_t_nz_r2']:.6f}"),
                ('NZ Tranverse Momentum Pearson r', f"{test_info['p_t_nz_pearson_r']:.6f}"),
                ('NZ Tranverse Momentum Pearson p', f"{test_info['p_t_nz_pearson_p']:.6f}"),
                ('NZ Tranverse Momentum Spearman r', f"{test_info['p_t_nz_spearman_r']:.6f}"),
                ('NZ Tranverse Momentum Spearman p', f"{test_info['p_t_nz_spearman_p']:.6f}"),
                ('Runtime', f"{test_info['run_time']}")
                )

        logging.info('\n'.join((
            '',
            center_text(f"Test hidden_size={msize}", ' '),
            table
        )))




    output_dir = os.path.join(config['output_dir'], 'summary')
    logging.info(f'Saving all to {output_dir}')
    logging.shutdown()


if __name__ == '__main__':
    main()
