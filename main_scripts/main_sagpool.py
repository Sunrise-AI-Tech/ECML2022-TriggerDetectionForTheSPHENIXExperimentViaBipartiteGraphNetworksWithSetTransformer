import random
import os
import sys
import argparse
import copy
import shutil
import logging
import yaml
import pickle
from datetime import datetime
from sklearn.metrics import roc_auc_score
from icecream import ic
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.SAGPool import SAGPoolNet
from models.ParticleNetLaplace import ParticleNetLaplace
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table

class ArgDict:
    pass

DEVICE = 'cuda:1'
OLD_COLUMNS = None

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/cluster_of_tracks_pns.yaml')
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
    argparser.add_argument('--use_wandb', action='store_true',
                        help="use wandb project name")
    argparser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    argparser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

    args = argparser.parse_args()

    return args

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

    return accum_info


def train(data, model_pnl, model, optimizer, epoch, output_dir, use_wandb=False,  threshold=0.5, use_radius=False):
    train_info = do_epoch(data, model_pnl, model, epoch, optimizer, threshold=threshold, use_radius=use_radius)
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model_pnl, model, epoch, loss_config=None, threshold=0.5, use_radius=False):
    with torch.no_grad():
        val_info = do_epoch(data, model_pnl, model, epoch, optimizer=None, threshold=threshold, use_radius=use_radius)
    return val_info


def do_epoch(data, model_pnl, model, epoch, optimizer=None, threshold=0.5, use_radius=False):
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
        'loss_contrast',
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
    preds_prob = []
    correct = []
    total_size = 0
    total_selected = 0
    
    for batch in data:
        tracks, vtx, partitions_as_graph, n_tracks, trig, energy, momentum, is_trigger_track, radii = batch
        tracks = tracks.to(DEVICE, torch.float)
        trig = (trig.to(DEVICE) == 1).long()
        vtx = vtx.to(DEVICE)
        batch_size = tracks.shape[0]
        num_insts += batch_size

        if use_radius:
            radii = radii.to(DEVICE)
            radii = radii.to(tracks.dtype)
            tracks = torch.cat((tracks, radii), dim=-1)


        if tracks.shape[0] == 1:
            skipped_batches += 1
            try:
                pred_x, edge_vals = model_pnl(tracks.transpose(-1, -2))
            except ValueError:
                continue
        else:
           pred_x, edge_vals = model_pnl(tracks.transpose(-1, -2))

        pred_x = pred_x.transpose(2, 1)


        data_x = torch.cat((tracks, pred_x), dim=-1)

        size_b, size_n, size_c = data_x.shape
        data_x = data_x.reshape(size_b * size_n, size_c)
        data_is_trigger_track = is_trigger_track.reshape(is_trigger_track.shape[0] * is_trigger_track.shape[1]).squeeze(-1)

        data_trig = trig
        if trig.size() == []:
            data_trig = trig.unsqueeze(-1)

        data_batch = torch.tensor(range(size_b)).unsqueeze(-1).repeat([1, size_n]).flatten()

        edge_vals = torch.nn.Sigmoid()(edge_vals)
        data_matrix_adj = (edge_vals >= threshold)
        data_edge_batch, data_edge_start, data_edge_end = (data_matrix_adj > 0).nonzero(as_tuple=False).t()
        data_edge_start += size_n * data_edge_batch
        data_edge_end += size_n * data_edge_batch
        data_edge_index = torch.stack([data_edge_start, data_edge_end])

        data_edge_attr = ((edge_vals + edge_vals.transpose(1, 2)) / 2)[data_matrix_adj].flatten().unsqueeze(-1)

        total_size += size_b * size_n * size_n
        total_selected += data_edge_index.shape[1]

        data_to_sagpool = torch_geometric.data.Data(
            x=data_x,
            edge_index=data_edge_index,
            y=data_trig,
            batch=data_batch,
            batch_size=size_b,
            edge_attr=data_edge_attr,
            ).to(DEVICE)
            
        loss = 0

        pred_labels = model(data_to_sagpool)
        ce_loss = F.nll_loss(pred_labels, data_to_sagpool.y)
        loss += ce_loss
        accum_info['loss_ce'] += ce_loss.item() * batch_size

        pred = pred_labels.max(dim=1)[1]
        preds.extend(pred.cpu().data.numpy())
        preds_prob.extend(nn.Softmax(dim=1)(pred_labels)[:, 1].detach().cpu().numpy().flatten())
        correct.extend(data_to_sagpool.y.detach().cpu().numpy().flatten())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(trig, pred_labels, accum_info)
        accum_info['loss'] += loss.item() * batch_size

    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    accum_info['loss'] /= num_insts
    accum_info['loss_ce'] /= num_insts
    accum_info['loss_contrast'] /= num_insts
    accum_info['ri'] = (tp + tn)/(tp + tn + fp + fn)
    accum_info['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
    accum_info['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
    accum_info['fscore'] = (2 * tp)/(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    correct = np.array(correct)
    preds = np.array(preds)
    preds_prob = np.array(preds_prob)

    try:
        accum_info['auroc'] = roc_auc_score(correct, preds_prob)
    except ValueError:
        accum_info['auroc'] = 0

    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)

    logging.info("\nThreshold: " + str(threshold) + ' Percentage: ' + str(total_selected / total_size))

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


    # Create model instance
    model_pnl = None
    if config['adj_model']['type'] == 'particlenet-lite-laplace':
        conv_params = (
            (config['adj_model']['k'], tuple(range(9)), (32, 32, 32), 'mean', 'ReLU'),
            (config['adj_model']['k'], tuple(range(32)), (64, 64, 64), 'mean', 'ReLU')
        )
        mlp_params = ((128, 0.1),)
        final_pooling = 'mean'
        input_dim = config['adj_model']['num_features'] + config['data']['use_radius'] + 13*config['data']['add_geo_features']
        model_pnl = ParticleNetLaplace(
                conv_params,
                mlp_params,
                final_pooling,
                input_dim,
                **config['adj_model']
        )
    elif config['adj_model']['type'] == 'particlenet-laplace':
        conv_params = (
                (config['adj_model']['k'], tuple(range(config['adj_model']['k'])), (64, 64, 64), 'mean', 'ReLU'),
                (config['adj_model']['k'], tuple(range(config['adj_model']['k'])), (128, 128, 128), 'mean', 'ReLU'),
                (config['adj_model']['k'], tuple(range(config['adj_model']['k'])), (64, 64, 64), 'mean', 'ReLU'),
        )
        mlp_params = ((256, 0.1),)
        final_pooling = 'mean'
        input_dim = config['adj_model']['num_features'] + config['data']['use_radius'] + 13*config['data']['add_geo_features']
        ic(input_dim)
        model_pnl = ParticleNetLaplace(
                conv_params,
                mlp_params,
                final_pooling,
                input_dim,
                **config['adj_model']
        )
    else:
        raise NotImplementedError(f'Model {config["adj_model"]["type"]} not implemented.')

    mconfig = copy.copy(config['model'])
    mconfig['num_features'] += config['data']['use_radius'] + 13*config['data']['add_geo_features'] + config['adj_model']['GNN_config']['hidden_dim']
    model = SAGPoolNet(
        **mconfig
    )
    model_pnl = model_pnl.to(DEVICE)
    model = model.to(DEVICE)

    checkpoint_file_pnl = config['checkpoint_file_particlenet']
    model_pnl = load_checkpoint(checkpoint_file_pnl, model_pnl)
    model_pnl.eval()


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
    best_val_ri = -1
    best_val_auroc = -1
    best_model = None
    
    for epoch in range(1, config['epochs'] + 1):
        train_info = train(train_data, model_pnl, model, optimizer, epoch, config['output_dir'], threshold=config['threshold'], use_radius=config['data']['use_radius'])
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

        val_info = evaluate(val_data, model_pnl, model, epoch, threshold=config['threshold'],
                use_radius=config['data']['use_radius']
                )
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

        if val_info['ri'] > best_val_ri:
            best_val_ri = val_info['ri']
            best_val_auroc = val_info['auroc']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        lr_scheduler.step()
        
    
    del train_data, val_data

    logging.info(f'Best validation acc: {best_val_ri:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    test_info = evaluate(test_data, model_pnl, best_model, epoch,
            use_radius=config['data']['use_radius']
        )

    table = make_table(
        ('Total loss', f"{test_info['loss']:.6f}"),
        ('Rand Index', f"{test_info['ri']:.6f}"),
        ('F-score', f"{test_info['fscore']:.4f}"),
        ('Recall', f"{test_info['recall']:.4f}"),
        ('Precision', f"{test_info['precision']:.4f}"),
        ('True Positives', f"{test_info['true_positives']}"),
        ('False Positives', f"{test_info['false_positives']}"),
        ('True Negatives', f"{test_info['true_negatives']}"),
        ('False Negatives', f"{test_info['false_negatives']}"),
        ('AUC Score', f"{test_info['auroc']:.6f}"),
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
        results_dict = {'train_loss': train_loss,
                        'train_ri': train_ri,
                        'val_loss': val_loss,
                        'val_ri': val_ri}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_ri': best_val_ri, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)

    logging.shutdown()



if __name__ == '__main__':
    main()
