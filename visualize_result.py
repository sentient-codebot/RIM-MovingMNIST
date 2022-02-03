"""Main entry point of the code"""
from __future__ import print_function

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from networks import BallModel
# from model_components import GruState 
from argument_parser import argument_parser
from dataset import get_dataloaders
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import plot_frames, plot_curve
from box import Box

import os
from os import listdir
from os.path import isfile, join

set_seed(0)

loss_fn = torch.nn.BCELoss()

def repackage_hidden(ten_):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(ten_, torch.Tensor):
        return ten_.detach()
    else:
        return tuple(repackage_hidden(v) for v in ten_)
      

def main():
    """Function to run the experiment"""
    args = argument_parser()
    args.id = f"SchemaBlocks_{args.hidden_size}_{args.num_units}"+\
        f"_{args.experiment_name}_{args.lr}_num_inp_heads_{args.num_input_heads}"+\
        f"_ver_{args.version}"
    # name="SchemaBlocks_"$dim1"_"$block1"_"$topk1"_"$something"_"$lr"_inp_heads_"$inp_heads"_templates_"$templates"_enc_"$encoder"_ver_"$version"_com_"$comm"_Sharing"
    print(args)
    logbook = LogBook(config=args)

    if not args.should_resume:
        # New Experiment
        make_dir(f"{args.folder_log}/model")
        logbook.write_message_logs(message=f"Saving args to {args.folder_log}/model/args")
        torch.save({"args": vars(args)}, f"{args.folder_log}/model/args")

    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    # args.device = torch.device("cpu")
    model = setup_model(args=args, logbook=logbook)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.directory = './data' # dataset directory
    # args.directory = 'D:\Projecten\Recurrent-Independent-Mechanisms\data' # dataset directory, windows os
    train_loader, test_loader, transfer_loader = get_dataloaders(args)

    train_batch_idx = 0

    start_epoch = 1
    for epoch in range(start_epoch,start_epoch+1):
        model.eval()
        with torch.no_grad():
            data = next(iter(train_loader))
            hidden = model.init_hidden(data.shape[0]).to(args.device) # NOTE initialize per epoch or per batch [??]
            data = data.to(args.device)
            hidden = hidden.detach()
            pred = torch.zeros_like(data)
            for frame in range(49):
                output, hidden = model(data[:, frame, :, :, :], hidden)
                pred[:,frame+1,:,:,:] = output
            
            pred = pred[:,1:,:,:,:]
            plot_frames(pred, data, 10, 20, 6)

def setup_model(args, logbook):
    """Method to setup the model"""

    model = BallModel(args)
    if args.should_resume:
        # Find the last checkpointed model and resume from that
        model_dir = f"{args.folder_log}/model"
        latest_model_idx = max(
            [int(model_idx) for model_idx in listdir(model_dir)
             if model_idx != "args"]
        )
        args.path_to_load_model = f"{model_dir}/{latest_model_idx}"
        args.checkpoint = {"epoch": latest_model_idx}
    else:
        assert False, 'set args.should_resume true!'

    return model


if __name__ == '__main__':
    main()
