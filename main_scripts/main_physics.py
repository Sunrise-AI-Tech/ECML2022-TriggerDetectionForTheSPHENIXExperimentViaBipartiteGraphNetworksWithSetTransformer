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
    global writer
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
        'r2',
        'e_mse',
        'e_mae',
        'e_r2',
        'e_x_mse',
        'e_x_mae',
        'e_y_mse',
        'e_x_r2',
        'e_y_mae',
        'e_y_r2',
        'e_z_mse',
        'e_z_mae',
        'e_z_r2',
        'p_mse', 
        'p_mae',
        'p_r2',
        'p_x_mse', 
        'p_x_mae',
        'p_x_r2',
        'p_y_mse', 
        'p_y_mae',
        'p_y_r2',
        'p_z_mse', 
        'p_z_mae',
        'p_z_r2',
        'p_t_mae',
        'p_t_mse',
        'p_t_r2'
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
    accum_info['e_mse'] /= num_insts
    accum_info['e_mae'] /= num_insts
    accum_info['e_x_mse'] /= num_insts
    accum_info['e_x_mae'] /= num_insts
    accum_info['e_y_mse'] /= num_insts
    accum_info['e_y_mae'] /= num_insts
    accum_info['e_z_mse'] /= num_insts
    accum_info['e_z_mae'] /= num_insts
    accum_info['p_mse'] /= num_insts
    accum_info['p_mae'] /= num_insts
    accum_info['p_x_mse'] /= num_insts
    accum_info['p_x_mae'] /= num_insts
    accum_info['p_y_mse'] /= num_insts
    accum_info['p_y_mae'] /= num_insts
    accum_info['p_z_mse'] /= num_insts
    accum_info['p_z_mae'] /= num_insts
    accum_info['p_t_mae'] /= num_insts
    accum_info['p_t_mse'] /= num_insts

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
        index += 1

    for m, v in accum_info.items():
        writer.add_scalar(f'{m}/{phase}', v, epoch)
           
    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)

    return accum_info

def main(auto=False, parser_dict=None, trails_number=None, datasets=None):
    global writer
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
    if mconfig['type'] == 'MLP':
        del mconfig['type']
        model = MLP(
                **mconfig
            )
    elif mconfig['type'] == 'HitsLSTM':
        del mconfig['type']
        model = HitsLSTM(
                **mconfig
            )
    else:
        raise NotImplementedError('Model {config["model"]["type"]} not implemented.')

    model = model.to(DEVICE)

    writer = SummaryWriter(log_dir=config['tensorboard_output_dir'])
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
    best_val_momentum_r2 = -1
    best_val_metrics = {}

    for epoch in range(1, config['epochs'] + 1):
        train_info = train(train_data, model, optimizer, epoch, config['output_dir'], loss=config['loss']['type'], use_energy=config['data']['use_energy'], use_momentum=config['data']['use_momentum'], use_pt=config['data']['use_pt'], use_radius=config['data']['use_radius'])
        table = make_table(
            ('Total loss', f"{train_info['loss']:.6f}"),
            ('Mean Squared Error', f"{train_info['mse']:.6f}"),
            ('Mean Absolute Error', f"{train_info['mae']:.6f}"),
            ('R^2', f"{train_info['r2']:.6f}"),
            ('Energy Mean Squared Error', f"{train_info['e_mse']:.6f}"),
            ('Energy Mean Absolute Error', f"{train_info['e_mae']:.6f}"),
            ('Energy R^2', f"{train_info['e_r2']:.6f}"),
            ('E_x Mean Squared Error', f"{train_info['e_x_mse']:.6f}"),
            ('E_x Mean Absolute Error', f"{train_info['e_x_mae']:.6f}"),
            ('E_x R^2', f"{train_info['e_x_r2']:.6f}"),
            ('E_y Mean Squared Error', f"{train_info['e_y_mse']:.6f}"),
            ('E_y Mean Absolute Error', f"{train_info['e_y_mae']:.6f}"),
            ('E_y R^2', f"{train_info['e_y_r2']:.6f}"),
            ('E_z Mean Squared Error', f"{train_info['e_z_mse']:.6f}"),
            ('E_z Mean Absolute Error', f"{train_info['e_z_mae']:.6f}"),
            ('E_z R^2', f"{train_info['e_z_r2']:.6f}"),
            ('Momentum Mean Squared Error', f"{train_info['p_mse']:.6f}"),
            ('Momentum Mean Absolute Error', f"{train_info['p_mae']:.6f}"),
            ('Momentum R^2', f"{train_info['p_r2']:.6f}"),
            ('p_x Mean Squared Error', f"{train_info['p_x_mse']:.6f}"),
            ('p_x Mean Absolute Error', f"{train_info['p_x_mae']:.6f}"),
            ('p_x R^2', f"{train_info['p_x_r2']:.6f}"),
            ('p_y Mean Squared Error', f"{train_info['p_y_mse']:.6f}"),
            ('p_y Mean Absolute Error', f"{train_info['p_y_mae']:.6f}"),
            ('p_y R^2', f"{train_info['p_y_r2']:.6f}"),
            ('p_z Mean Squared Error', f"{train_info['p_z_mse']:.6f}"),
            ('p_z Mean Absolute Error', f"{train_info['p_z_mae']:.6f}"),
            ('p_z R^2', f"{train_info['p_z_r2']:.6f}"),
            ('Tranverse Momentum MAE', f"{train_info['p_t_mae']}"),
            ('Tranverse Momentum MSE', f"{train_info['p_t_mse']}"),
            ('Tranverse Momentum R^2', f"{train_info['p_t_r2']:.6f}"),
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
                use_energy=config['data']['use_energy'],
                use_momentum=config['data']['use_momentum'],
                use_pt=config['data']['use_pt'],
                use_radius=config['data']['use_radius'],
                )

        table = make_table(
            ('Total loss', f"{val_info['loss']:.6f}"),
            ('Mean Squared Error', f"{val_info['mse']:.6f}"),
            ('Mean Absolute Error', f"{val_info['mae']:.6f}"),
            ('R^2', f"{val_info['r2']:.6f}"),
            ('Energy Mean Squared Error', f"{val_info['e_mse']:.6f}"),
            ('Energy Mean Absolute Error', f"{val_info['e_mae']:.6f}"),
            ('Energy R^2', f"{val_info['e_r2']:.6f}"),
            ('E_x Mean Squared Error', f"{val_info['e_x_mse']:.6f}"),
            ('E_x Mean Absolute Error', f"{val_info['e_x_mae']:.6f}"),
            ('E_x R^2', f"{val_info['e_x_r2']:.6f}"),
            ('E_y Mean Squared Error', f"{val_info['e_y_mse']:.6f}"),
            ('E_y Mean Absolute Error', f"{val_info['e_y_mae']:.6f}"),
            ('E_y R^2', f"{val_info['e_y_r2']:.6f}"),
            ('E_z Mean Squared Error', f"{val_info['e_z_mse']:.6f}"),
            ('E_z Mean Absolute Error', f"{val_info['e_z_mae']:.6f}"),
            ('E_z R^2', f"{val_info['e_z_r2']:.6f}"),
            ('Momentum Mean Squared Error', f"{val_info['p_mse']:.6f}"),
            ('Momentum Mean Absolute Error', f"{val_info['p_mae']:.6f}"),
            ('Momentum R^2', f"{val_info['p_r2']:.6f}"),
            ('p_x Mean Squared Error', f"{val_info['p_x_mse']:.6f}"),
            ('p_x Mean Absolute Error', f"{val_info['p_x_mae']:.6f}"),
            ('p_x R^2', f"{val_info['p_x_r2']:.6f}"),
            ('p_y Mean Squared Error', f"{val_info['p_y_mse']:.6f}"),
            ('p_y Mean Absolute Error', f"{val_info['p_y_mae']:.6f}"),
            ('p_y R^2', f"{val_info['p_y_r2']:.6f}"),
            ('p_z Mean Squared Error', f"{val_info['p_z_mse']:.6f}"),
            ('p_z Mean Absolute Error', f"{val_info['p_z_mae']:.6f}"),
            ('p_z R^2', f"{val_info['p_z_r2']:.6f}"),
            ('Tranverse Momentum MAE', f"{val_info['p_t_mae']}"),
            ('Tranverse Momentum MSE', f"{val_info['p_t_mse']}"),
            ('Tranverse Momentum R^2', f"{val_info['p_t_r2']:.6f}"),
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
    test_info = evaluate(test_data, best_model, epoch,
            loss=config['loss']['type'],
            use_energy=config['data']['use_energy'],
            use_momentum=config['data']['use_momentum'],
            use_pt=config['data']['use_pt'],
            use_radius=config['data']['use_radius'],
            )

    table = make_table(
            ('Total loss', f"{test_info['loss']:.6f}"),
            ('Mean Squared Error', f"{test_info['mse']:.6f}"),
            ('Mean Absolute Error', f"{test_info['mae']:.6f}"),
            ('R^2', f"{test_info['r2']:.6f}"),
            ('Energy Mean Squared Error', f"{test_info['e_mse']:.6f}"),
            ('Energy Mean Absolute Error', f"{test_info['e_mae']:.6f}"),
            ('Energy R^2', f"{test_info['e_r2']:.6f}"),
            ('E_x Mean Squared Error', f"{test_info['e_x_mse']:.6f}"),
            ('E_x Mean Absolute Error', f"{test_info['e_x_mae']:.6f}"),
            ('E_x R^2', f"{test_info['e_x_r2']:.6f}"),
            ('E_y Mean Squared Error', f"{test_info['e_y_mse']:.6f}"),
            ('E_y Mean Absolute Error', f"{test_info['e_y_mae']:.6f}"),
            ('E_y R^2', f"{test_info['e_y_r2']:.6f}"),
            ('E_z Mean Squared Error', f"{test_info['e_z_mse']:.6f}"),
            ('E_z Mean Absolute Error', f"{test_info['e_z_mae']:.6f}"),
            ('E_z R^2', f"{test_info['e_z_r2']:.6f}"),
            ('Momentum Mean Squared Error', f"{test_info['p_mse']:.6f}"),
            ('Momentum Mean Absolute Error', f"{test_info['p_mae']:.6f}"),
            ('Momentum R^2', f"{test_info['p_r2']:.6f}"),
            ('p_x Mean Squared Error', f"{test_info['p_x_mse']:.6f}"),
            ('p_x Mean Absolute Error', f"{test_info['p_x_mae']:.6f}"),
            ('p_x R^2', f"{test_info['p_x_r2']:.6f}"),
            ('p_y Mean Squared Error', f"{test_info['p_y_mse']:.6f}"),
            ('p_y Mean Absolute Error', f"{test_info['p_y_mae']:.6f}"),
            ('p_y R^2', f"{test_info['p_y_r2']:.6f}"),
            ('p_z Mean Squared Error', f"{test_info['p_z_mse']:.6f}"),
            ('p_z Mean Absolute Error', f"{test_info['p_z_mae']:.6f}"),
            ('p_z R^2', f"{test_info['p_z_r2']:.6f}"),
            ('Tranverse Momentum MAE', f"{test_info['p_t_mae']}"),
            ('Tranverse Momentum MSE', f"{test_info['p_t_mse']}"),
            ('Tranverse Momentum R^2', f"{test_info['p_t_r2']:.6f}"),
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

    hparams_metrics = {}
    for k, v in best_val_metrics.items():
        if k == 'run_time':
            continue

        hparams_metrics[f'{k}/validation'] = v

    writer.add_hparams(hp, hparams_metrics,
    hparam_domain_discrete = {
        'type/optimizer': ['Adam', 'SGD'],
        'load_complete_graph/data': [True, False],
    })

    writer.close()

    if auto:
        return best_val_ri

    logging.shutdown()


if __name__ == '__main__':
    main()
