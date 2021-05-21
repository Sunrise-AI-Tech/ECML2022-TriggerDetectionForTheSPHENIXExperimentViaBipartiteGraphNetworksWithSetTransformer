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
import sklearn.metrics as metrics
from scipy.sparse import coo_matrix

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn


"""
How To:
Example for running from command line:
python <path_to>/SetToGraph/main_scripts/main_jets.py
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
# from dataloaders import jets_loader
# from performance_eval.eval_test_jets import eval_jets_on_test_set
from dataloaders.hits_loader import get_data_loaders
from utils import write_checkpoint

DEVICE = 'cuda'

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('config', nargs='?', default='configs/GNNDiffpool.yaml')
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('--auto', action='store_true')
    # argparser.add_argument('-e', '--epochs', default=400, type=int, help='The number of epochs to run')
    # argparser.add_argument('-l', '--lr', default=0.0003, type=float, help='The learning rate')
    # argparser.add_argument('-b', '--bs', default=50, type=int, help='Batch size to use')
    # argparser.add_argument('--method', default='lin2', help='Method to transfer from sets to graphs: lin2 for S2G, lin5 for S2G+')
    # argparser.add_argument('--res_dir', default='../experiments/jets_results', help='Results directory')
    # argparser.add_argument('--baseline', default=None, help='Use a baseline and not set2graph. mlp, gnn, siam or siam3.')

    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--show-config', action='store_true')
    argparser.add_argument('--resume', action='store_true', default=0, help='Resume from last checkpoint')
    
    args = argparser.parse_args()

    # assert args.baseline is None or args.baseline in ['mlp', 'gnn', 'siam', 'siam3']

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


def do_epoch(phase, data, models, epoch, output_dir=None):
    if phase == 'train':
        for model in models:
            model.train()
        if epoch > 1:
            for model in models:
                model.scheduler.step()
    else:
        for model in models:
            model.eval()
    start_time = datetime.now()
    accum_info = {k: 0.0 for k in ['ri', 'loss', 'insts', 'accuracy', 'fscore', 'precision', 'recall', 'label_acc', 'CEloss', 'Glaploss', 'lr']}
    accum_info['lr'] = models[0].scheduler.get_last_lr()[0]
    preds = []
    labels = []
    event_labels = []
    count = 0
    for batch in data:
        count += 1
        # print(batch)
        batch_size = batch.batch[-1]+1
        hits = batch.hits #hits_info. : (N1+ N2 + ...)*input_features 
        edge_index = batch.edge_index #GNN-edge
        hits_to_track = batch.hits_to_track #for scatter add hits to tracks
        # partitions = batch.tracks_label #tracks label, which track belongs to which vtx (0 is pv) B*N
        # partitions_as_graph = batch.partition_as_graph #groundtruth adjacent matrix, B*N*N
        # weight_matrix = batch.weight_matrix #weight adjacent matrix, B*N*N
        # vtx = batch.track_vtx #coordinate of vertex for each track, B*N*3
        # vtxm = batch.track_vtxm
        trig = batch.trigger #trigger or not, 0 for NN, 1 for trigger, 2 for ND
        event_labels.append(trig.long())
        # trigger_flag represents trigger or not
        trig = (trig == 1)
        trig = trig.to(DEVICE, torch.float)
        labels.append(trig.long().cpu().numpy())
        n_hits = batch.n_hits # number of nodes
        hits_cumsum = np.cumsum(n_hits)
        n_tracks = batch.n_tracks # number of tracks
        # ip_true = batch.ip

        # print(f'hits shape: {hits.shape}')
        # print(f'partition shape: {partitions.shape}')
        # print(f'hits_to_track: {hits_to_track}')
        # print(f'n_hits: {n_hits}')
        # print(f'batch.batch: {batch.batch}')
        

        # One Train step on the current batch
        hits = hits.to(DEVICE, torch.float)
        edge_index = edge_index.to(DEVICE, torch.long)
        hits_to_track = hits_to_track.to(DEVICE, torch.long)
        # partitions = partitions.to(DEVICE, torch.float)
        # partitions_as_graph = partitions_as_graph.to(DEVICE, torch.float)
        # weight_matrix = weight_matrix.to(DEVICE, torch.float)
        # vtx= vtx.to(DEVICE, torch.float)
        # vtxm = vtxm.to(DEVICE, torch.float)
        # event_labels.append(trig.long())
        hits_cumsum = hits_cumsum.to(DEVICE, torch.long)
        # ip_true = ip_true.to(DEVICE, torch.float)
        
        #resize matrix

        # vtx = vtx.view(batch_size, n_tracks[0], 3)
        # vtxm = vtxm.view(batch_size, n_tracks[0], 3)
        # ip_true = ip_true.view(batch_size, 3).to(DEVICE, torch.float)
        batch.batch = batch.batch.to(DEVICE, torch.long)
        # partitions = partitions.view(batch_size, n_tracks[0])
        # partitions_as_graph = partitions_as_graph.view(batch_size, n_tracks[0], n_tracks[0])
        # weight_matrix = weight_matrix.view(batch_size, n_tracks[0], n_tracks[0])


        accum_info['insts'] += batch_size

        with torch.set_grad_enabled(phase == 'train'):
            if model.name == 'GNNPairDiffpool':
                ip_pred = model(hits, edge_index, batch.batch, batch_size, hits_to_track, hits_cumsum, n_tracks[0])
            else:
                ip_pred = model(hits, edge_index, batch.batch, batch_size)
            ip_pred = ip_pred.squeeze(1)
            preds.append((ip_pred>0).cpu().data.numpy())     
            if phase == 'train':
                loss = model.train_model(ip_pred, trig)
            else:
                loss = model.get_loss(ip_pred, trig)
                

        # update results from train_step func
        accum_info['loss'] += loss.item() * batch_size

    num_insts = accum_info.pop('insts')
    accum_info['loss'] /= num_insts
    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    event_labels = np.hstack(event_labels)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
                'recall': metrics.recall_score(labels, preds, average='macro'),
                'acc': metrics.accuracy_score(labels, preds),
                'F1': metrics.f1_score(labels, preds, average="micro")}
    accum_info['label_acc'] = metrics.accuracy_score(labels, preds)
    
    #logging
    if phase == 'train':
        logging.info(f"\tTraining - {epoch:4}")
        write_checkpoint(checkpoint_id=epoch, model=model, optimizer=model.optimizer, learning_rate=model.learning_rate, output_dir=output_dir)
        # model.scheduler.step()
    else:
        logging.info(f"\tVal - {epoch:4}")
    logging.info(f'event classification result: {result}')
    logging.info(f'event classification confusion matrix: {metrics.confusion_matrix(labels, preds)}')
    sample_weight = np.ones(preds.shape[0], dtype=np.int64)
    logging.info(f'bigger confusion matrix: {coo_matrix((sample_weight, (event_labels, preds)), shape=(3, 2), dtype=np.int64).toarray()}')
    return accum_info

def main(auto=False, parser_dict=None, trails_number=None, datasets=None, config_file=None):
    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

     # Parse the command line
    args = parse_args()
    print(args)

    # Load configuration
    config = load_config(args.config)
    # print(config)
    if auto:
        config = config_file
        config['output_dir'] = parser_dict['output_dir']
        config['loss_fun'] = parser_dict['loss_fun']
        loss_weights = parser_dict['loss_fun']
    
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    file_handler = config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=0)
    if auto:
        logging.info('---------------------------------------------------------------')
        logging.info(f'-------------------------trail_num: {trails_number}--------------------------')
        logging.info(f'loss_weight: {loss_weights}')
        logging.info('---------------------------------------------------------------')
    logging.info('Command line config: %s' % args)
    logging.info('Configuration: %s', config)
    logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    save_config(config)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.cuda.set_device(int(args.gpu))

    if config['model_name'] == 'GNN_Diffpool_ensemble':
        from models.GNN_diffpool import GNNDiffpool
        model0 = GNNDiffpool(**config['model'])
        model1 = GNNDiffpool(**config['model'])
    else:
        logging.info('Wrong model!')
        return
        
    
    print_model_summary(model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The nubmer of model parameters is {num_params}')

    if auto:
        train_data, val_data = datasets
    else:
        # Load data
        logging.info('Loading training data and validation data')
        # data_argument = {'input_dir': 'physics_data', 'n_train': 80, 
        # 'n_valid': 20, 'real_weight': 3, 'batch_size': 100, 'n_workers': 32, 'n_input_dir': 1,
        # 'input_dir2': 'results/gcoord_ND_100k', 'input_dir3': 'results/gcoord_ND_100k'}
        train_data, val_data = get_data_loaders(**config['data'], model_name=model.name)
    logging.info('Loaded %g training samples', len(train_data.dataset))
    logging.info('Loaded %g validation samples', len(val_data.dataset))

    # hidden_dim = config['hidden_dim']

    # Metrics
    train_loss = np.empty(config['epochs'], float)
    val_loss = np.empty(config['epochs'], float)

    best_epoch = -1
    # best_val_ri = -1
    best_val_loss = 10000
    # best_model = None

    
    #train_data: batch -> list with len 4 : sets (track_info, b*n*16, 16=3*3 hits+2 edge length+edge angle+track length+hits center), partitions (track_label,b*n), partitions_as_graph (affinity matrix, ground truth, b*n*n), weight_matrix(b*n*n) 
#    for i, batch in enumerate(train_data):
#        print(i)
#        for itm in batch:
#          print(itm.shape)
    
    # Training and evaluation process
    for epoch in range(1, config['epochs'] + 1):
        train_info = do_epoch('train', train_data, [model0, model1], epoch, config['output_dir'])
        logging.info(f"total loss:{train_info['loss']:.6f}  --acc: {train_info['label_acc']:.4f} -- runtime:{train_info['run_time']} --lr: {train_info['lr']:.6f}")
        train_loss[epoch-1] = train_info['loss']

        val_info = do_epoch('valid', val_data, [model0, model1], epoch)
        logging.info(f"total loss:{val_info['loss']:.6f}  --acc: {val_info['label_acc']:.4f} -- runtime:{val_info['run_time']}\n \
 #######################################################################################################")
        val_loss[epoch-1] = val_info['loss']

        if val_info['loss'] < best_val_loss:
            best_val_loss = val_info['loss']
            best_epoch = epoch
            # best_model = copy.deepcopy(model)

        # lr_scheduler.step()
        
        # if best_epoch < epoch - 20:
        #     print('Early stopping training due to no improvement over the last 20 epochs...')
        #     break

    del train_data, val_data
    logging.info(f'Best validation loss: {best_val_loss:.4f}, best epoch: {best_epoch}.')

    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    # Saving to disk
    if args.save:
        exp_dir = f'jets_{start_time:%Y%m%d_%H%M%S}_0'
        output_dir = os.path.join(config['output_dir'], exp_dir)

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
        # torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'code.py'))
        results_dict = {'train_loss': train_loss,
                        'val_loss': val_loss}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_loss': best_val_loss, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)

    # print('Loading test data...', end='', flush=True)
    # test_data = jets_loader.get_data_loader('test', config.bs, config.debug_load)
    # test_info = evaluate(test_data, best_model)
    # print(f"\tTest     - {best_epoch:4}",
    #       " loss:{loss:.6f} -- mean_ri:{ri:.4f} -- fscore:{fscore:.4f} -- recall:{recall:.4f} "
    #       "-- precision:{precision:.4f}  -- runtime:{run_time}\n".format(**test_info))

    logging.info(f'Epoch {best_epoch} - evaluating over test set.')

    logging.info(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')

    if auto:
        return best_val_acc

    logging.shutdown()


if __name__ == '__main__':
    main()