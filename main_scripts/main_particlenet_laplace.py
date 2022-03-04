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
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.ParticleNetLaplace import ParticleNetLaplace
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table
from torch.utils.tensorboard import SummaryWriter

class ArgDict:
    pass

writer = None
DEVICE = 'cuda:1'
def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/cluster_of_tracks_particlenet_laplace.yaml')
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('--auto', action='store_true')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--show-config', action='store_true')
    argparser.add_argument('--resume', action='store_true', default=0, help='Resume from last checkpoint')
    
    # Logging
    argparser.add_argument('--name', type=str, default=None,
            help="Run name")

    args = argparser.parse_args()

    return args

def load_config(config_file, **kwargs):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Update config from command line, and expand paths
    config['output_dir'] = os.path.expandvars(config['output_dir'])
    for key, val in kwargs.items():
        config[key] = val
    return config

def config_logging(verbose, output_dir, append=False, rank=0):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if output_dir is not None:
        log_dir = output_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'out_%i.log' % rank)
        mode = 'a' if append else 'w'
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)
    # Suppress annoying matplotlib debug printouts
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    return file_handler

def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

def print_model_summary(model):
    """Override as needed"""
    logging.info(
        'Model: \n%s\nParameters: %i' %
        (model, sum(p.numel() for p in model.parameters())))
    logging.info(
        'Model: \n%s\Buffers: %i' %
        (model, sum([buf.nelement()*buf.element_size() for buf in model.buffers() if buf != 'Pairwise_Predictor'])))
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers() if buf != 'Pairwise_Predictor'])
    logging.info(mem_bufs)
    mem = mem_params + mem_bufs

def calc_metrics(trig, pred, accum_info):
    with torch.no_grad():
        assert len(pred.shape) == 2
        pred = torch.softmax(pred, dim=-1)
        tp = torch.sum((trig == 1) * (torch.argmax(pred, dim=-1) == 1)).item()
        tn = torch.sum((trig == 0) * (torch.argmax(pred, dim=-1) == 0)).item()
        fp = torch.sum((trig == 0) * (torch.argmax(pred, dim=-1) == 1)).item()
        fn = torch.sum((trig == 1) * (torch.argmax(pred, dim=-1) == 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

        accum_info['ri'] += (tp + tn)/(tp + tn + fp + fn)

    return accum_info


def calc_metrics_for_linkage(truth, pred, accum_info):
    with torch.no_grad():
        assert len(pred.shape) == 3
        pred = torch.nn.Sigmoid()(pred)
        
        pred_binary = pred>0.5

        tp = torch.sum((truth == 1) * (pred_binary == 1)).item()
        tn = torch.sum((truth == 0) * (pred_binary == 0)).item()
        fp = torch.sum((truth == 0) * (pred_binary == 1)).item()
        fn = torch.sum((truth == 1) * (pred_binary == 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

    return accum_info

def extract_hyperparameters(config):
    hp = {}
    hp['epochs'] = config['epochs']
    hp['type/optimizer'] = config['optimizer']['type']
    hp['momentum/optimizer'] = config['optimizer']['momentum']
    hp['weight_decay/optimizer'] = config['optimizer']['weight_decay']
    hp['learning_rate/optimizer'] = config['optimizer']['learning_rate']
    hp['type/model'] = config['model']['type']
    hp['hidden_dim/model'] = config['model']['hidden_dim']
    hp['hidden_activation/model'] = config['model']['hidden_activation']
    hp['layer_norm/model'] = config['model']['layer_norm']
    hp['affinity_loss/model'] = config['model']['affinity_loss']
    hp['affinity_loss_CE_weight/model'] = config['model']['affinity_loss_CE_weight']
    hp['affinity_loss_Lp_weight/model'] = config['model']['affinity_loss_Lp_weight']
    hp['affinity_loss_11_weight/model'] = config['model']['affinity_loss_11_weight']
    hp['affinity_loss_frobenius_weight/model'] = config['model']['affinity_loss_frobenius_weight']
    hp['d_metric/model'] = config['model']['d_metric']
    hp['k/model'] = config['model']['k']
    hp['hidden_dim/GNN_config/model'] = config['model']['GNN_config']['hidden_dim']
    hp['hidden_activation/GNN_config/model'] = config['model']['GNN_config']['hidden_activation']
    hp['layer_norm/GNN_config/model'] = config['model']['GNN_config']['layer_norm']
    hp['n_graph_iters/GNN_config/model'] = config['model']['GNN_config']['n_graph_iters']
    hp['name/data'] = config['data']['name']
    hp['n_train/data'] = config['data']['n_train']
    hp['n_valid/data'] = config['data']['n_valid']
    hp['batch_size/data'] = config['data']['batch_size']
    hp['load_complete_graph/data'] = config['data']['load_complete_graph']
    hp['use_momentum/data'] = config['data']['use_momentum']
    hp['use_energy/data'] = config['data']['use_energy']
    hp['use_radius/data'] = config['data']['use_radius']

    return hp



'''
Input:
@vtx    [batch_size, n_tracks, 3]           3 stands for 3d coordinates

Output:
@res    [batch_size, n_tracks, n_tracks]    Connectivity Truth
'''
def get_A_from_vtx(vtx):
    # ic(vtx.shape)
    res = torch.zeros((vtx.shape[0], vtx.shape[1], vtx.shape[1]))
    # ic(res.shape)
    i_batch = 0
    for node_coords in vtx:
        for i in range(len(node_coords)):
            for j in range(len(node_coords)):
                if torch.equal(node_coords[i], node_coords[j]):
                    res[i_batch][i][j] = 1
        i_batch = i_batch + 1
    # ic(res)
    return res

def train(data, model, optimizer, epoch,  output_dir, use_energy=False, use_momentum=False, use_radius=False):
    train_info = do_epoch(data, model, epoch, optimizer=optimizer, use_energy=use_energy, use_momentum=use_momentum, use_radius=use_radius)
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model, epoch, use_energy=False, use_momentum=False, use_radius=False):
    with torch.no_grad():
        val_info = do_epoch(data, model, epoch, optimizer=None, use_energy=use_energy, use_momentum=use_momentum, use_radius=use_radius)
    return val_info

def do_epoch(data, model, epoch, optimizer=None, use_energy=False, use_momentum=False, use_radius=False):
    global writer
    if optimizer is None:
        # validation epoch
        model.eval()
        phase = 'validation'
    else:
        # train epoch
        model.train()
        phase = 'train'

    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in (
        'ri', 
        'auroc', 
        'loss', 
        'loss_ce', 
        'loss_lp', 
        'loss_frobenius', 
        'loss_11', 
        'fscore', 
        'precision', 
        'recall', 
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives'
    )}

    num_insts = 0
    skipped_batches = 0
    preds = []
    correct = []
    for batch in data:
        tracks, vtx, partitions_as_graph, n_tracks, trig, energy, momentum, is_trigger_track, radii = batch
        tracks = tracks.float().to(DEVICE)
        if use_radius:
            radii = radii.to(DEVICE)
            radii = radii.to(tracks.dtype)
            tracks = torch.cat((tracks, radii), dim=-1)

        tracks = tracks.transpose(-1, -2)
        trig = (trig.to(DEVICE) == 1).long()
        vtx= vtx.to(DEVICE)

        batch_size = tracks.shape[0]
        num_insts += batch_size
        
        if tracks.shape[0] == 1:
            skipped_batches += 1
            try:
               pred_x, pred_A = model(tracks)
            except ValueError:
                continue
        else:
           pred_x, pred_A = model(tracks)

        
        partitions_as_graph = partitions_as_graph.view(batch_size, n_tracks[0], n_tracks[0])
        partitions_as_graph = partitions_as_graph.to(DEVICE, torch.float)

        A_true = partitions_as_graph

        preds.extend((torch.nn.Sigmoid()(pred_A) > 0.5).detach().float().cpu().numpy().flatten())
        correct.extend(partitions_as_graph.detach().cpu().numpy().flatten())

        loss, loss_ce, loss_lp, loss_frobenius, loss_11 = model.get_loss(pred_A, A_true, vtx)

        if optimizer is not None:
            # backprop for training epochs only
            optimizer.zero_grad()
            (loss/batch_size).backward()
            optimizer.step()

        # calc ri
        # accum_info = calc_metrics(trig, pred, accum_info)
        accum_info = calc_metrics_for_linkage(A_true, pred_A, accum_info)

        # update results from train_step func
        accum_info['loss'] += loss.item()
        accum_info['loss_ce'] += loss_ce.item()
        accum_info['loss_lp'] += loss_lp.item()
        accum_info['loss_frobenius'] += loss_frobenius.item()
        accum_info['loss_11'] += loss_11.item()

    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    accum_info['loss'] /= num_insts
    accum_info['loss_ce'] /= num_insts
    accum_info['loss_lp'] /= num_insts
    accum_info['loss_frobenius'] /= num_insts
    accum_info['loss_11'] /= num_insts
    accum_info['ri'] = (tp + tn)/(tp + tn + fp + fn)
    accum_info['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
    accum_info['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
    accum_info['fscore'] = (2 * tp)/(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    correct = np.array(correct)
    preds = np.array(preds)

    accum_info['auroc'] = roc_auc_score(correct, preds)

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
    if auto:
        config['tensorboard_output_dir'] = parser_dict['tensorboard_output_dir']

    hp = extract_hyperparameters(config)
    
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
        del dconfig['use_energy']
        del dconfig['use_radius']
        train_data, val_data = get_data_loaders(**dconfig)
        logging.info('Loaded %g training samples', len(train_data.dataset))
        logging.info('Loaded %g validation samples', len(val_data.dataset))

    # Create model instance
    if config['model']['type'] == 'particlenet-lite-laplace':
        conv_params = (
            (config['model']['k'], tuple(range(9)), (32, 32, 32), 'mean', 'ReLU'),
            (config['model']['k'], tuple(range(32)), (64, 64, 64), 'mean', 'ReLU')
        )
        mlp_params = ((128, 0.1),)
        final_pooling = 'mean'
        input_dim = 28 + config['data']['use_radius']
        model = ParticleNetLaplace(
                conv_params,
                mlp_params,
                final_pooling,
                input_dim,
                **config['model']
        )
    elif config['model']['type'] == 'particlenet-laplace':
        conv_params = (
                (config['model']['k'], tuple(range(config['model']['k'])), (64, 64, 64), 'mean', 'ReLU'),
                (config['model']['k'], tuple(range(config['model']['k'])), (128, 128, 128), 'mean', 'ReLU'),
                (config['model']['k'], tuple(range(config['model']['k'])), (64, 64, 64), 'mean', 'ReLU'),
        )
        mlp_params = ((256, 0.1),)
        final_pooling = 'mean'
        input_dim = 28 + config['data']['use_radius']
        model = ParticleNetLaplace(
                conv_params,
                mlp_params,
                final_pooling,
                input_dim,
                **config['model']
        )
    else:
        raise NotImplementedError(f'Model {config["model"]["type"]} not implemented.')

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
    train_loss = np.empty(config['epochs'], float)
    train_ri = np.empty(config['epochs'], float)
    val_loss = np.empty(config['epochs'], float)
    val_ri = np.empty(config['epochs'], float)

    best_epoch = -1
    best_val_auroc = -1
    best_val_ri = -1
    best_model = None
    
    for epoch in range(1, config['epochs'] + 1):
        train_info = train(
                train_data, 
                model, 
                optimizer, 
                epoch, 
                config['output_dir'],
                use_energy=config['data']['use_energy'], 
                use_momentum=config['data']['use_momentum'],
                use_radius=config['data']['use_radius']
                )
        table = make_table(
            ('Total loss', f"{train_info['loss']:.6f}"),
            ('Rand Index', f"{train_info['ri']:.6f}"),
            ('F-score', f"{train_info['fscore']:.4f}"),
            ('Recall', f"{train_info['recall']:.4f}"),
            ('Precision', f"{train_info['precision']:.4f}"),
            ('True Positives', f"{train_info['true_positives']}"),
            ('False Positives', f"{train_info['false_positives']}"),
            ('True Negatives', f"{train_info['true_negatives']}"),
            ('False Negatives', f"{train_info['false_negatives']}"),
            ('AUC Score', f"{train_info['auroc']:.6f}"),
            ('Runtime', f"{train_info['run_time']}")
        )
        logging.info('\n'.join((
            '',
            "#" * get_terminal_columns(),
            center_text(f"Training - {epoch:4}", ' '),
            table
        )))
        train_loss[epoch-1], train_ri[epoch-1] = train_info['loss'], train_info['ri']

        val_info = evaluate(val_data, model, epoch, 
                use_energy=config['data']['use_energy'], 
                use_momentum=config['data']['use_momentum'],
                use_radius=config['data']['use_radius'])
        table = make_table(
            ('Total loss', f"{val_info['loss']:.6f}"),
            ('Rand Index', f"{val_info['ri']:.6f}"),
            ('F-score', f"{val_info['fscore']:.4f}"),
            ('Recall', f"{val_info['recall']:.4f}"),
            ('Precision', f"{val_info['precision']:.4f}"),
            ('True Positives', f"{val_info['true_positives']}"),
            ('False Positives', f"{val_info['false_positives']}"),
            ('True Negatives', f"{val_info['true_negatives']}"),
            ('False Negatives', f"{val_info['false_negatives']}"),
            ('AUC Score', f"{val_info['auroc']:.6f}"),
            ('Runtime', f"{val_info['run_time']}")
        )
        logging.info('\n'.join((
            '',
            center_text(f"Validation - {epoch:4}", ' '),
            table
            )))

        val_loss[epoch-1], val_ri[epoch-1] = val_info['loss'], val_info['ri']


        if val_info['auroc'] > best_val_auroc:
            best_val_auroc = val_info['auroc']
            best_val_ri = val_info['ri']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        lr_scheduler.step()
        
    
    del train_data, val_data
    logging.info(f'Best validation acc: {best_val_ri:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

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
        results_dict = {'train_loss': train_loss,
                        'train_ri': train_ri,
                        'val_loss': val_loss,
                        'val_ri': val_ri}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_ri': best_val_ri, 'best_val_ri': best_val_ri, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)


    writer.add_hparams(hp, {
        'auroc/validation': best_val_auroc,
        'ri/validation': best_val_ri,
    },
    hparam_domain_discrete = {
        'd_metric/model': ['intertrack', 'einsum'],
        'type/model': ['particlenet-laplace', 'particlenet-lite-laplace'],
        'type/optimizer': ['Adam', 'SGD'],
        'load_complete_graph/data': [True, False]

    })
    writer.close()
    if auto:
        return best_val_ri

    
    logging.shutdown()

if __name__ == '__main__':
    main()
