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
from icecream import ic
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
from models.MLP import MLP
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

DEVICE = 'cuda:1'
OLD_COLUMNS = None

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/gt_track_physics.yaml')
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

def train(data, model, optimizer, epoch, output_dir, loss='mse', use_radius=False):
    train_info = do_epoch(data, model, epoch, optimizer, loss=loss, use_radius=use_radius) 
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model, epoch, loss='mse', use_radius=False):
    with torch.no_grad():
        val_info = do_epoch(data, model, epoch, optimizer=None, use_radius=use_radius) 
    return val_info

def do_epoch(data, model, epoch, optimizer=None, loss='mse', use_radius=False):
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
        gts.append(pts)
            
        physics_true = torch.cat(gts, dim=-1)
        pt = physics_true.detach().cpu().numpy()
        pt = np.reshape(pt, (-1, pt.shape[-1]))
        correct.extend(pt)

        losses = []
        pp = physics_pred[..., 0:1]
        pt = physics_true[..., 0:1]
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
    accum_info['p_t_r2'] = r2[0]
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

    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)

    return accum_info



def main():
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
    
    config['output_dir'] = os.path.join(config['output_dir'], f'experiment_{start_time:%Y-%m-%d_%H:%M:%S}')
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    file_handler = config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=0)
    logging.info('Command line config: %s' % args)
    logging.info('Configuration: %s', config)
    logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    save_config(config)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.cuda.set_device(int(args.gpu))

    # Load data
    logging.info('Loading training data and validation data')
    dconfig = copy.copy(config['data'])

    train_data, val_data, test_data = get_data_loaders(**dconfig)
    logging.info('Loaded %g training samples', len(train_data.dataset))
    logging.info('Loaded %g validation samples', len(val_data.dataset))
    logging.info('Loaded %g test samples', len(test_data.dataset))


    mconfig = copy.copy(config['model'])
    mconfig['n_classes'] = 1
    mconfig['n_features'] += 13*config['data']['add_geo_features'] + config['data']['use_radius']
    model = MLP(
            **mconfig
        )
    model = model.to(DEVICE)

    # Optimizer
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['learning_rate'], weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config['optimizer']['learning_rate'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {config["optimizer"]["type"]} not implemented.')


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

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print_model_summary(model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The number of model parameters is {num_params}')

    # Metrics
    train_metrics = defaultdict(lambda: np.zeros(config['epochs']))
    val_metrics = defaultdict(lambda: np.zeros(config['epochs']))

    best_epoch = -1
    best_val_momentum_r2 = -np.inf
    best_val_metrics = {}

    for epoch in range(1, config['epochs'] + 1):
        train_info = evaluate(train_data, model, epoch,
                loss=config['loss']['type'],
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

        for k, v in train_info.items():
            if k == 'run_time':
                continue

            train_metrics[k][epoch-1] = v
 
        logging.info('\n'.join((
            '',
            "#" * get_terminal_columns(),
            center_text(f"Training - {epoch:4}", ' '),
            table
        )))

        val_info = evaluate(val_data, model, epoch,
                loss=config['loss']['type'],
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
            center_text(f"Validation - {epoch:4}", ' '),
            table
        )))


        for k, v in val_info.items():
            if k == 'run_time':
                continue

            val_metrics[k][epoch-1] = v
 
 
        if val_info['p_t_r2'] > best_val_momentum_r2:
            best_val_momentum_r2 = val_info['p_t_r2']
            best_val_metrics = copy.copy(val_info)
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        lr_scheduler.step()
        
    
    del train_data, val_data

    logging.info(f'Best validation momentum R^2: {best_val_momentum_r2:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    test_info = evaluate(test_data, model, epoch,
            loss=config['loss']['type'],
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
        center_text(f"Test", ' '),
        table
        )))



    # Saving to disk
    if args.save:
        output_dir = os.path.join(config['output_dir'], 'summary')
        i = 0
        while True:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)  # raises error if dir already exists
                break
            i += 1
            output_dir = output_dir[:-1] + str(i)
            if i > 9:
                logging.info(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        logging.info(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'code.py'))
        results_dict = {}
        for k, v in train_metrics.items():
            results_dict[f'train_{k}'] = v
        for k, v in val_metrics.items():
            results_dict[f'val_{k}'] = v
                        
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = copy.copy(best_val_metrics)
        best_dict['loss'] = config['loss']['type']
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)

    logging.shutdown()


if __name__ == '__main__':
    main()
