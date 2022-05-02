"""Utility functions"""
import collections
import os
import random
import pathlib

import numpy as np
import torch


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def flatten_dict(d, parent_key='', sep='#'):
    """Method to flatten a given dict using the given seperator.
    Taken from https://stackoverflow.com/a/6027615/1353861
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_dir(path):
    """Make dir"""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def distinctiveness(m1, m2, v1, v2):
    dist = torch.sum((m1-m2)**2)/(v1+v2)
    coef = torch.exp(-dist)
    return coef

  

def slot_loss(slot_means, slot_variances):
    loss = 0.0
    for k1 in range(slot_means.shape[1]):
        for k2 in range(k1+1,slot_means.shape[1]):
            loss += distinctiveness(slot_means[:,k1],slot_means[:,k2],
                                  slot_variances[:,k1],slot_variances[:,k2])
    return loss

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor

def variance_loss(dist):
    coef = torch.exp(-dist)
    return coef    

def build_grid2D(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    # st()

    return torch.from_numpy(grid)

def distinctiveness_2(m1, m2):
    dist = torch.exp(-torch.abs((m1-m2)))
    return dist


def slot_distinctiveness_2(slot_means):
    loss = 0.0
    for k1 in range(slot_means.shape[1]):
        for k2 in range(k1+1,slot_means.shape[1]):
            loss += distinctiveness_2(slot_means[:,k1],slot_means[:,k2])
    return loss
