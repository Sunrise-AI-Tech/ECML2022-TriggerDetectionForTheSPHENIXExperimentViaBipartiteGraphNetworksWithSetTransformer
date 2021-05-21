import os
import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def load_config(config_file, **kwargs):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Update config from command line, and expand paths
    config['output_dir'] = os.path.expandvars(config['output_dir'])
    for key, val in kwargs.items():
        config[key] = val
    return config

def print_model_summary(model):
    """Override as needed"""
    logging.info(
        'Model: \n%s\nParameters: %i' %
        (model, sum(p.numel() for p in model.parameters())))


def write_checkpoint(checkpoint_id, model, optimizer, learning_rate, output_dir):
    """Write a checkpoint for the model"""
    model_state_dict = model.state_dict()
    checkpoint = dict(checkpoint_id=checkpoint_id,
                        model=model_state_dict,
                        optimizer=optimizer.state_dict(),
                        learning_rate=learning_rate)
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

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.1) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])