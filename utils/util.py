"""Utility functions"""
import collections
import os
import random
import pathlib
import math
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed, strict=False):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if strict:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' # or ":16:2"


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
    ranges = [np.linspace(0., res-1, num=res) for res in resolution]
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

    
def is_nan(tensor: Union[torch.Tensor,float]) -> bool:
    if isinstance(tensor, torch.Tensor):
        return bool(torch.isnan(tensor).any())
    else:
        return math.isnan(tensor)
    
def load_model(
    model: torch.nn.Module,
    model_dir: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    curr_epoch: int = 1000,
) -> int:
    """
    resume checkpoint. so far only used for recovering from failed epoch. 
    
    return (int) the checkpoint epoch
    """
    _checkpoint_list = [int(f.split('.')[0]) for f in os.listdir(model_dir) if f.endswith('.pt')]
    checkpoint_list = [item for item in _checkpoint_list if item <= curr_epoch] # don't loaf "future" checkpoints!
    if len(checkpoint_list) == 0:
        raise RuntimeError('Training loss anomaly. Failed when trying to reload checkpoint: No Checkpoint Found.')
    else:
        print(
            "Training loss anomaly. Trying to resume previous checkpoint. "
        )
        latest_model_idx = max(checkpoint_list)
        path = os.path.join(f"{model_dir}", f"{latest_model_idx}.pt")
        checkpoint = torch.load(path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print(
            f"Resuming from epoch {start_epoch-1}"
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint resumed")
        
        return start_epoch - 1
    
class LMSFilter():
    """
    assume  y = h@x + n
            n ~ N(0, var)
    """
    def __init__(self, p, bias=True):
        self.h = torch.zeros(p+1, 1) if bias else torch.zeros(p, 1)
        self.var = torch.ones(1)
        self.bias = bias
        
    def to(self, str):
        self.h = self.h.to(str)
    
    def infer(self, x):
        return self.forward(x)
    
    def learn(self, x, y):
        # y: Shape [N, 1]
        # x: Shape [N, p]
        pred = self.forward(x) # [N, 1]
        if self.bias:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            x = torch.cat((ones, x), dim=1)
        delta = ((y - pred) * x / x.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True).t()
        self.var = 0.9*self.var + 0.1*(y-pred).square().mean()
        self.h = self.h + 0.1*delta
        return pred
        
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [N, p]

        Returns:
            _type_: _description_
        """
        if not self.bias:
            return torch.matmul(x, self.h)
        else:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            x = torch.cat((ones, x), dim=1)
            return torch.matmul(x, self.h)
class AnomalyDetector():
    """class definition for an anomaly detector"""
    def __init__(self, init_len=25, p=5):
        self.record = []
        self.count = 0 # count for recorded datapoints
        self.init_len = init_len
        self.max_len = p
        self.lms = LMSFilter(p=5, bias=False)
        
    def put(self, datapoint: torch.Tensor) -> None:
        datapoint = self._to_tensor(datapoint)
        if len(self.record) < self.max_len:
            self.record.append(datapoint)
        else:
            self.record.pop(0)
            self.record.append(datapoint)
        self.count += 1
            
    def _to_tensor(self, datapoint):
        if not isinstance(datapoint, torch.Tensor):
            datapoint = torch.tensor(datapoint, dtype=torch.float)
        return datapoint
    
    def __call__(self, datapoint = torch.Tensor) -> bool:
        datapoint = self._to_tensor(datapoint)
        
        if is_nan(datapoint):
            return True
        if len(self.record) >= self.max_len:
            past = torch.stack(self.record, dim=0).view(1,-1) # [1, p]
            if self.count > self.init_len:
                var = self.lms.var
                pred = self.lms.infer(past)
                if datapoint > pred + 5.*var.sqrt():
                    return True # anomaly
            self.lms.learn(past, datapoint)
        self.put(datapoint)
        return False # normal
    