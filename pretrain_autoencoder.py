from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd
from torch.utils.tensorboard import SummaryWriter

from networks import BallModel, SlotAttentionAutoEncoder
from argument_parser import argument_parser
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import ScalarLog, VectorLog, HeatmapLog
from utils.metric import f1_score
import utils.pssim.pytorch_ssim as pt_ssim
from datasets import setup_dataloader
from tqdm import tqdm
from test_mmnist import dec_rim_util, test

import os 
from os import listdir
from os.path import isfile, join

print("This process has the PID: ", os.getpid())

set_seed(1997)

def train(model, train_loader, optimizer, epoch, train_batch_idx, args, loss_fn):
    model.train()

    train_epoch_loss = torch.tensor(0.).to(args.device)
    for batch_idx, data in enumerate(tqdm(train_loader)):
        labels, in_frames, out_frames = [tensor.to(args.device) for tensor in data]
        data = torch.cat((in_frames, out_frames), dim=1) # Shape; [N, T, C, H, W]
        start_time = time()
        data = data.to(args.device)
        optimizer.zero_grad()

        loss = 0.
        for frame in range(data.shape[1]):
            output = model(data[:, frame, :, :, :])
            target = data[:, frame, :, :, :]
            loss += loss_fn(output, target)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True) 
        optimizer.step()

        train_batch_idx += 1 
        train_epoch_loss = train_epoch_loss.detach() + loss.detach()

    train_epoch_loss = train_epoch_loss / (batch_idx+1)

    return train_batch_idx, train_epoch_loss

def test(model, test_loader, args, loss_fn):
    model.eval()
    mse = torch.nn.MSELoss()

    epoch_loss = torch.tensor(0.).to(args.device)
    epoch_mseloss = torch.tensor(0.).to(args.device)
    f1 = 0.
    ssim = 0.
    for batch_idx, data in enumerate(test_loader): # tqdm doesn't work here?
        labels, in_frames, out_frames = [tensor.to(args.device) for tensor in data]
        data = torch.cat((in_frames, out_frames), dim=1) # Shape; [N, T, C, H, W]
        data = data.to(args.device)
        if data.dim()==4:
            data = data.unsqueeze(2).float()
        loss = 0.
        mseloss = 0.
        prediction = torch.zeros_like(data)

        for frame in range(data.shape[1]):
            with torch.no_grad():
                output = model(data[:, frame, :, :, :])
                target = data[:, frame, :, :, :]
                prediction[:, frame, :, :, :] = output

                loss += loss_fn(output, target)
                mseloss += mse(output, target)
                f1_frame = f1_score(target, output)
                f1 += f1_frame

        ssim += pt_ssim.ssim(data[:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])), # data.shape = (batch, frame, 1, height, width)
                            prediction[:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])))
        epoch_loss += loss.detach()
        epoch_mseloss += mseloss.detach()

    epoch_loss = epoch_loss / (batch_idx+1)
    epoch_mseloss = epoch_mseloss / (batch_idx+1)
    ssim = ssim / (batch_idx+1)
    f1_avg = f1 / (batch_idx+1) / (data.shape[1]-1)

    metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg
    }

    return epoch_loss, metrics

def main():
    args = argument_parser()

    if not args.should_resume:
        make_dir(f"{args.folder_save}/pretrain/checkpoints")
        make_dir(f"{args.folder_save}/pretrain/args")
        torch.save({
            "args": vars(args)
        }, f"{args.folder_save}/pretrain/args/args")

    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    model, optimizer, start_epoch, train_batch_idx = setup_model(args=args)

    train_loader, test_loader = setup_dataloader(args=args)
    transfer_loader = test_loader

    if args.loss_fn == "BCE":
        loss_fn = torch.nn.BCELoss() 
    elif args.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif args.loss_fn == 'MAE':
        loss_fn = torch.nn.L1Loss()
    else:
        loss_fn = torch.nn.MSELoss()

    writer = SummaryWriter(log_dir='./runs/pretrain/'+args.id)

    for epoch in range(start_epoch, args.epochs+1):
        train_batch_idx, epoch_loss = train(
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            epoch = epoch,
            train_batch_idx = train_batch_idx,
            args = args,
            loss_fn = loss_fn,
        )

        # test done here
        writer.add_scalar('Loss/Train Loss '+f'({args.loss_fn.upper()})', epoch_loss.detach(), epoch)
        if args.test_frequency > 0 and epoch % args.test_frequency == 0 or epoch <= 15:
            test_loss, metrics = test(
                model = model,
                test_loader = test_loader,
                args = args,
                loss_fn = loss_fn,
            )
            writer.add_scalar('Loss/Test Loss '+f'({args.loss_fn.upper()})', test_loss.detach(), epoch)
            for key, val in metrics.items():
                writer.add_scalar(f'Metrics/{key}', val, epoch)
            print(f"Epoch {epoch} | Train Loss {epoch_loss.detach():.4f} | Test Loss {test_loss.detach():.4f} | "+\
                f"Test MSE: {metrics['mse']:.4f} | "+\
                f"Test F1 score: {metrics['f1']:.4f} | Test SSIM: {metrics['ssim']:.4f}")
        else:
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f}")

        # save checkpoints here
        if args.save_frequency > 0 and epoch % args.save_frequency == 0 or epoch==1: # regularly save checkpoints
            print(f"Saving model to {args.folder_save}/pretrain/checkpoints/{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"{args.folder_save}/pretrain/checkpoints/{epoch}.pt")

    writer.close()
        
def setup_model(args):
    model = SlotAttentionAutoEncoder(
        input_size=args.input_size,
        num_iterations=args.num_iterations_slot,
        num_slots=args.num_slots,
        slot_size=args.slot_size,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    train_batch_idx = 0
    if args.should_resume:
        # Find the last checkpointed model and resume from that
        model_dir = f"{args.folder_log}/pretrain/checkpoints"
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
    
        print(f"Resuming experiment id: {args.id}, from epoch: {start_epoch}")

    return model, optimizer, start_epoch, train_batch_idx

if __name__ == '__main__':
    main()


