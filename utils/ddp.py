import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datasets import setup_dataloader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS
from torch.utils.data import DataLoader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
def setup_dataloader_dist(rank, world_size, args, pin_memory=False, num_worker=0):
    train_set, val_set, test_set = setup_dataloader(args, return_dataset=True)
    sampler = DS(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size//world_size,
                              pin_memory=pin_memory,
                              num_workers=num_worker,
                              sampler=sampler
                              )
    test_loader = DataLoader(test_set, batch_size=args.batch_size//2, 
                             pin_memory=pin_memory,
                             num_workers=2,
                             shuffle=False,
    )    
    return train_loader, None, test_loader