from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd

from networks import BallModel
from argument_parser import argument_parser
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import ScalarLog, VectorLog, HeatmapLog
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

def train(model, train_loader, optimizer, epoch, logbook, train_batch_idx, args):
    grad_norm_log = ScalarLog(args.folder_log, "grad_norm", epoch=epoch)

    model.train()

    epoch_loss = torch.tensor(0.).to(args.device)
    for batch_idx, data in enumerate(train_loader):
        hidden = model.init_hidden(data.shape[0]).to(args.device)

        start_time = time()
        data = data.to(args.device)
        data = data.unsqueeze(2).float()
        hidden = hidden.detach()
        optimizer.zero_grad()
        loss = 0.
        # with autograd.detect_anomaly():
        if True:
            for frame in range(data.shape[1]-1):
                output, hidden = model(data[:, frame, :, :, :], hidden)

                nan_hook(output)
                nan_hook(hidden)
                target = data[:, frame+1, :, :, :]
                loss += loss_fn(output, target)
                
            loss.backward()
            grad_norm = get_grad_norm(model)
            grad_norm_log.append(grad_norm)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True) 
            optimizer.step()
            grad_norm_log.save()

        train_batch_idx += 1 
        metrics = {
            "loss": loss.cpu().item(),
            "mode": "train",
            "batch_idx": train_batch_idx,
            "epoch": epoch,
            "time_taken": time() - start_time,
        }
        logbook.write_metric_logs(metrics=metrics)

        epoch_loss += loss.detach()

    if args.log_intm_frequency > 0 and epoch % args.log_intm_frequency == 0:
        """log intermediate variables here"""
        pass
        # SAVE logged vectors

    epoch_loss = epoch_loss / (batch_idx+1)
    return train_batch_idx, epoch_loss

def main():
    args = argument_parser()

    print(args)
    logbook = LogBook(config = args)

    if not args.should_resume:
        make_dir(f"{args.folder_log}/checkpoints")
        make_dir(f"{args.folder_log}/model")
        logbook.write_message_logs(message=f"Saving args to {args.folder_log}/model/args")
        torch.save({
            "args": vars(args)
        }, f"{args.folder_log}/model/args")

    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    model, optimizer, start_epoch, train_batch_idx, epoch_loss_log = setup_model(args=args, logbook=logbook)

    train_set = MovingMNIST(root='./data', train=True, download=True)
    test_set = MovingMNIST(root='./data', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    transfer_loader = test_loader

    for epoch in range(start_epoch, args.epochs+1):
        train_batch_idx, epoch_loss = train(
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            epoch = epoch,
            logbook = logbook,
            train_batch_idx = train_batch_idx,
            args = args
        )
        epoch_loss_log.append(epoch_loss)
        epoch_loss_log.save()

        # no test done here

        if args.model_persist_frequency > 0 and epoch % args.model_persist_frequency == 0:
            logbook.write_message_logs(message=f"Saving model to {args.folder_log}/checkpoints/{epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'epoch_loss_log': epoch_loss_log
            }, f"{args.folder_log}/checkpoints/{epoch}")
        
def setup_model(args, logbook):
    model = BallModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    train_batch_idx = 0
    epoch_loss_log = ScalarLog(args.folder_log+'/intermediate_vars', "epoch_loss")
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['epoch']
        epoch_loss_log = checkpoint['epoch_loss_log']
    
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id}, from epoch: {start_epoch}")

    return model, optimizer, start_epoch, train_batch_idx, epoch_loss_log

if __name__ == '__main__':
    main()


