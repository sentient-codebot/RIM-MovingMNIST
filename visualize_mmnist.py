from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd

from networks import BallModel
from argument_parser import argument_parser
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import plot_frames, plot_curve
from data.MovingMNIST import MovingMNIST
from box import Box

import os 
from os import listdir
from os.path import isfile, join

set_seed(1997)

loss_fn = torch.nn.BCELoss()

def nan_hook(_tensor):
        nan_mask = torch.isnan(_tensor)
        if nan_mask.any():
            raise RuntimeError(f"Found NAN in: ", nan_mask.nonzero(), "where:", _tensor[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def get_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

@torch.no_grad()
def test(model, test_loader, args):

    model.eval()

    epoch_loss = torch.tensor(0.).to(args.device)
    for batch_idx, data in enumerate(test_loader):
        hidden = model.init_hidden(data.shape[0]).to(args.device)

        start_time = time()
        data = data.to(args.device)
        data = data.unsqueeze(2).float()
        hidden = hidden.detach()
        loss = 0.
        prediction = torch.zeros_like(data)

        for frame in range(data.shape[1]-1):
            output, hidden, *_ = model(data[:, frame, :, :, :], hidden)

            nan_hook(output)
            nan_hook(hidden)
            target = data[:, frame+1, :, :, :]
            prediction[:, frame+1, :, :, :] = output
            loss += loss_fn(output, target)

        epoch_loss += loss.detach()
        break
        
    prediction = prediction[:, 1:, :, :, :]
    epoch_loss = epoch_loss / (batch_idx+1)
    return epoch_loss, prediction, data

def main():
    args = argument_parser()

    print(args)
    logbook = LogBook(config = args)

    if not args.should_resume:
        raise RuntimeError("args.should_resume should be set True. ")

    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    model = setup_model(args=args, logbook=logbook)

    test_set = MovingMNIST(root='./data', train=False, download=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    transfer_loader = test_loader
    
    epoch_loss, prediction, target = test(
        model = model,
        test_loader = test_loader,
        args = args
    )
    plot_frames(prediction, target, start_frame=0, end_frame=18, batch_idx=7)

        
def setup_model(args, logbook):
    model = BallModel(args)
    
    if args.should_resume:
        # Find the last checkpointed model and resume from that
        model_dir = f"{args.folder_log}/checkpoints"
        latest_model_idx = max(
            [int(model_idx) for model_idx in listdir(model_dir)
             if model_idx != "args"]
        )
        args.path_to_load_model = f"{model_dir}/{latest_model_idx}"
        args.checkpoint = {"epoch": latest_model_idx}

    if args.path_to_load_model != "":
        checkpoint = torch.load(args.path_to_load_model.strip())
        model.load_state_dict(checkpoint['model_state_dict'])
    
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id} from epoch: {args.checkpoint}")

    return model

if __name__ == '__main__':
    main()


