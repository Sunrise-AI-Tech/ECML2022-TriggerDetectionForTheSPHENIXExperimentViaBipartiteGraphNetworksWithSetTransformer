import os
import sys
import torch
import logging
import yaml
import pickle                                           

OLD_COLUMNS = None

def write_checkpoint(checkpoint_id, model, optimizer, output_dir):
    """Write a checkpoint for the model"""
    model_state_dict = model.state_dict()
    checkpoint = dict(checkpoint_id=checkpoint_id,
                        model=model_state_dict,
                        optimizer=optimizer.state_dict(),
                        )
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
    torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_file))

def load_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    return model

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

def calc_metrics(y, pred, accum_info):
    with torch.no_grad():
        assert len(pred.shape) == 2
        pred = torch.softmax(pred, dim=-1)
        tp = torch.sum((y == 1) * (torch.argmax(pred, dim=-1) == 1)).item()
        tn = torch.sum((y == 0) * (torch.argmax(pred, dim=-1) == 0)).item()
        fp = torch.sum((y == 0) * (torch.argmax(pred, dim=-1) == 1)).item()
        fn = torch.sum((y == 1) * (torch.argmax(pred, dim=-1) == 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

    return accum_info

def get_terminal_columns():
    # global OLD_COLUMNS
    # try:
    #     _, columns = os.popen('stty size', 'r').read().split()
    #     columns = int(columns)
    #     if OLD_COLUMNS is None:
    #         OLD_COLUMNS = columns
    # except:
    #     columns = 80
    # print(columns)
    # return min(columns, OLD_COLUMNS)
    return  80

def center_text(text, fill):
    columns = get_terminal_columns()
    left = max(len(text) - columns // 2, 0)
    right = max(columns - left, 0)
    return f'{" " * left}{text}{" " * right}'

def make_table(*lv_pairs):
    labels = tuple(map(lambda x: x[0], lv_pairs))
    values = tuple(map(lambda x: x[1], lv_pairs))
    values = tuple(map(str, values))

    max_label_length = max(map(len, labels))
    max_value_length = max(map(len, values))

    columns = get_terminal_columns()
    # +4 handles ' = ' at the end of label and ' ' at the end of value
    lv_pairs_per_row = columns // (max_label_length + max_value_length + 4)
    lv_pairs_per_row = max(lv_pairs_per_row, 1)

    label_format_string = ''.join(('{0:<', str(max_label_length), '} = '))
    value_format_string = ''.join(('{1:>', str(max_value_length), '} '))
    lv_pair_format_string = ''.join((label_format_string, value_format_string))
    

    table = []
    for i, (label, value) in enumerate(lv_pairs):
        if i > 0 and i % lv_pairs_per_row == 0:
            table.append('\n')
        table.append(lv_pair_format_string.format(label, value))

    return ''.join(table)

def get_performance_table(accum_info, len_data):
    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    accum_info['loss'] /= len_data
    accum_info['ri'] = (tp + tn)/(tp + tn + fp + fn)
    accum_info['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
    accum_info['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
    accum_info['fscore'] = (2 * tp)/(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0

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

        accum_info['accuracy'] += (tp + tn)/(tp + tn + fp + fn)
        accum_info['ri'] += (tp + tn)/(tp + tn + fp + fn)
#        accum_info['auroc'] += roc_auc_score(trig.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())

    return accum_info


def calc_metrics_for_linkage(truth, pred, accum_info):
    with torch.no_grad():
        assert len(pred.shape) == 3
        pred = torch.nn.Sigmoid()(pred)
        # ic(truth, truth.shape)
        # ic(pred, pred.shape)
        
        pred_binary = pred>0.5

        tp = torch.sum((truth == 1) * (pred_binary == 1)).item()
        tn = torch.sum((truth == 0) * (pred_binary == 0)).item()
        fp = torch.sum((truth == 0) * (pred_binary == 1)).item()
        fn = torch.sum((truth == 1) * (pred_binary == 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

#        accum_info['auroc'] += roc_auc_score(trig.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
    return accum_info
