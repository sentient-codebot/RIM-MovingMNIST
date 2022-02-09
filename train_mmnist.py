from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd

from networks import BallModel
from argument_parser import argument_parser
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import ScalarLog, VectorLog
from utils.visualize import HeatmapLog
from data.MovingMNIST import MovingMNIST
from box import Box
from tqdm import tqdm

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
    intm_log_folder = args.folder_log + '/intermediate_vars'
    inp_key_log = HeatmapLog(intm_log_folder, "inp key mat")
    inp_value_log = HeatmapLog(intm_log_folder, "inp value mat")
    inp_query_log = HeatmapLog(intm_log_folder, "inp query mat")
    comm_key_log = HeatmapLog(intm_log_folder, "comm key mat")
    comm_value_log = HeatmapLog(intm_log_folder, "comm value mat")
    comm_query_log = HeatmapLog(intm_log_folder, "comm query mat")

    grad_norm_log = ScalarLog(intm_log_folder, "grad_norm")
    encoded_log = VectorLog(intm_log_folder, "encoded", epoch=epoch) # TODO put them in a folder
    attn_score_log = VectorLog(intm_log_folder, "attn_score", epoch=epoch)
    hidden_log = VectorLog(intm_log_folder, "hidden_state", epoch=epoch)

    model.train()

    epoch_loss = torch.tensor(0.).to(args.device)
    for batch_idx, data in enumerate(tqdm(train_loader)):
        attn_score_log.reset()
        encoded_log.reset()
        hidden_log.reset()

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
                output, hidden, inp_ctx, comm_ctx, encoded_input = model(data[:, frame, :, :, :], hidden)
                # ----- logging -----
                encoded_log.append(encoded_input[-1]) # only take the last sample in a batch
                attn_score_cat = torch.cat(
                    (inp_ctx[3][-1].flatten(),comm_ctx[3][-1].flatten())
                )
                attn_score_log.append(attn_score_cat)
                hidden_log.append(hidden[-1].flatten())
                # ----- ------- -----
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

    if args.log_intm_frequency > 0 and epoch % args.log_intm_frequency == 0:
        """log intermediate variables here"""
        pass
        # TODO plot inp_ctx 
        inp_key_log.plot(inp_ctx[0], epoch) # happens in the first batch
        inp_value_log.plot(inp_ctx[1], epoch)
        inp_query_log.plot(inp_ctx[2], epoch)
        pass
        # TODO plot comm_ctx
        comm_key_log.plot(comm_ctx[0], epoch)
        comm_value_log.plot(comm_ctx[1], epoch)
        comm_query_log.plot(comm_ctx[2], epoch)
        # TODO SAVE logged vectors
        encoded_log.save()
        attn_score_log.save()
        hidden_log.save()

        epoch_loss += loss.detach()
        
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

    model, optimizer, start_epoch, train_batch_idx = setup_model(args=args, logbook=logbook)

    train_set = MovingMNIST(root='./data', train=True, download=True, mini=False)
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
    epoch_loss_log = ScalarLog(args.folder_log+'/intermediate_vars', "epoch_loss")
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
            }, f"{args.folder_log}/checkpoints/{epoch}")
        
def setup_model(args, logbook):
    model = BallModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    train_batch_idx = 0
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
    
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id}, from epoch: {start_epoch}")

    return model, optimizer, start_epoch, train_batch_idx

if __name__ == '__main__':
    main()


