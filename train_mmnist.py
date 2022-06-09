from multiprocessing.sharedctypes import Value
from tabnanny import check
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import wandb
from utils import util
from networks import BallModel, SlotAttentionAutoEncoder, TrafficModel
from argument_parser import argument_parser
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import make_grid_video
from utils.logging import log_stats, setup_wandb_columns
from datasets import setup_dataloader
from tqdm import tqdm
from test_mmnist import dec_rim_util, test

import os 
from os import listdir
from os.path import isfile, join

print("Python Process PID: ", os.getpid())

set_seed(1997, strict=True)

PRETRAINED_MODEL_PATH = './saves/PRETRAIN_MMNIST_SLOT_SA_3_100_3_RIM_6_100_ver_0/pretrain/checkpoints/encoder_sa.pt'

def get_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2).item() if p.grad is not None else 0.
        # param_norm = p.grad.detach().data.norm(2).item() 
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train(model, train_loader, optimizer, epoch, train_batch_idx, args, loss_fn, writer):
    model.train()

    epoch_loss = torch.tensor(0.).to(args.device)
    epoch_recon_loss = 0.
    epoch_pred_loss = 0.
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # data: (labels, frames_in, frames_out)
        if args.task == 'MMNIST':
            digit_labels, in_frames, out_frames = [tensor.to(args.device) for tensor in data] 
            data = torch.cat((in_frames, out_frames), dim=1) # [N, *T, 1, H, W]
        else:
            data = data.to(args.device)
        hidden = model.init_hidden(data.shape[0]).to(args.device)
        hidden = hidden.detach()
        memory = None
        if args.use_sw:
            memory = model.init_memory(data.shape[0]).to(args.device)
        
        optimizer.zero_grad()
        recon_loss = 0.
        pred_loss = 0.
        loss = 0.
        for frame in range(data.shape[1]-1):
            if args.spotlight_bias:
                recons, preds, hidden, memory, slot_means, slot_variances, attn_param_bias = model(data[:, frame, :, :, :], hidden, memory)
                target = data[:, frame+1, :, :, :]
                loss = loss + loss_fn(preds, target) + 0.1*torch.sum(util.slot_loss(slot_means,slot_variances)) + 0.01*torch.sum(attn_param_bias**2)
            else:
                recons, preds, hidden, memory = model(data[:, frame, :, :, :], hidden, memory)
                if recons is not None:
                    curr_target = data[:, frame, :, :, :]
                    recon_loss = recon_loss + loss_fn(recons, curr_target)
                next_target = data[:, frame+1, :, :, :]
                pred_loss = pred_loss + loss_fn(preds, next_target)
                loss = args.recon_loss_weight*recon_loss + (1.-args.recon_loss_weight)*pred_loss
            
        loss.backward()
        grad_norm = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False) 
        optimizer.step()
        writer.add_scalar('Grad Norm', grad_norm, train_batch_idx)

        train_batch_idx += 1 
        epoch_loss = epoch_loss + loss.detach()
        epoch_recon_loss += recon_loss.detach() if isinstance(recon_loss, torch.Tensor) else recon_loss
        epoch_pred_loss += pred_loss.detach() if isinstance(pred_loss, torch.Tensor) else pred_loss

    epoch_loss = epoch_loss / len(train_loader)
    epoch_recon_loss /= len(train_loader)
    epoch_pred_loss /= len(train_loader)

    return train_batch_idx, epoch_loss, epoch_recon_loss, epoch_pred_loss

def main():
    # parse and process args
    args = argument_parser()
    # print(args)
    cudable = torch.cuda.is_available()
    if cudable:
        args.device = torch.device("cuda")
    else:
        try:
            import torch.backends.mps as mps
            args.device = torch.device("cpu" if mps.is_available() else "cpu")
        except ModuleNotFoundError:
            args.device = torch.device("cpu")
    print(f'using device {args.device}')
    make_dir(args.folder_log)
    make_dir(f"{args.folder_save}/checkpoints")
    make_dir(f"{args.folder_save}/best_model")
    make_dir(f"{args.folder_save}/args")
    if not args.should_resume:
        print(f"Saving args to {args.folder_save}/args/args.pt")
        torch.save({
            "args": vars(args)
        }, f"{args.folder_save}/args/args.pt")

    # wandb setup
    project, name = args.id.split('_',1)
    wandb.init(project=project, name=name, config=vars(args), entity='nan-team', settings=wandb.Settings(start_method="thread"))
    columns = setup_wandb_columns(args) # artifact columns

    # data setup
    train_loader, _, test_loader = setup_dataloader(args=args)

    # model setup
    model, optimizer, scheduler, loss_fn, start_epoch, train_batch_idx, best_mse = setup_model(args=args)
    
    # tensorboard setup
    writer = SummaryWriter(log_dir='./runs/'+args.id)

    # training loop
    for epoch in range(start_epoch, args.epochs+1):
        # train 
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        train_batch_idx, train_loss, train_recon_loss, train_pred_loss = train(
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            epoch = epoch,
            train_batch_idx = train_batch_idx,
            args = args,
            loss_fn = loss_fn,
            writer = writer
        )
        loss_dict = {
            "train loss": train_loss.item(),
            "train recon loss": train_recon_loss.item() if isinstance(train_recon_loss, torch.Tensor) else train_recon_loss,
            "train pred loss": train_pred_loss.item() if isinstance(train_pred_loss, torch.Tensor) else train_pred_loss,
        }
        metric_dict = {
        }
        # scheduler.step(...)
        # scheduler.step(epoch) # NOTE disable for now

        # test 
        if args.test_frequency > 0 and epoch % args.test_frequency == 0 or epoch <= 15:
            """test model accuracy and log intermediate variables here"""
            test_loss, test_recon_loss, test_pred_loss, prediction, data, metrics, test_table = test(
                model = model, 
                test_loader = test_loader, 
                args = args, 
                loss_fn = loss_fn, 
                writer = writer,
                rollout = True,
                epoch = epoch,
                log_columns=columns if epoch%50==0 else None,
            )
            log_stats(
                args=args,
                is_train=True,
                epoch=epoch,
                train_loss=train_loss,
                train_recon_loss=train_recon_loss,
                train_pred_loss=train_pred_loss,
                test_loss=test_loss,
                test_recon_loss=test_recon_loss,
                test_pred_loss=test_pred_loss,
                ground_truth=data,
                prediction=prediction,
                metrics=metrics,
                test_table=test_table,
                writer=writer,
                lr=optimizer.param_groups[0]['lr'],
                manual_init_scale=0. if not args.use_past_slots else torch.sigmoid(model.slot_attention.manual_init_scale_digit.detach())
            )
            # save if better than bese
            loss_dict['test loss'] = test_loss
            if metrics['mse'] < best_mse:
                best_mse = metrics['mse']
                print(f'best test MSE: {best_mse:.4f}')
                if epoch > 10:
                    print(f"Saving model to {args.folder_save}/best_model/best.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_batch_idx': train_batch_idx,
                        'loss': test_loss,
                        'mse': metrics['mse'],
                        'best_mse': best_mse,
                    }, f"{args.folder_save}/best_model/best.pt")
                wandb.run.summary['best_mse'] = best_mse
                wandb.run.summary['best_mse_epoch'] = epoch
                wandb.run.summary['best_mse_f1'] = metrics.get('f1', -1)
                wandb.run.summary['best_mse_ssim'] = metrics.get('ssim', -1)
                wandb.run.summary.update()

        else:
            print(f"epoch {epoch}/{args.epochs} | "+\
                f"train loss: {train_loss:.4f}"
            )
            wandb.log({
                'Loss': loss_dict,
                'Stats': {
                    'Learning Rate': optimizer.param_groups[0]['lr'],
                    'Past Slot Init Scale': 0. if not args.use_past_slots else torch.sigmoid(model.slot_attention.manual_init_scale_digit).detach()},
            }, step=epoch)
        writer.add_scalars(f'Loss/{args.loss_fn.upper()}', 
            loss_dict, 
            epoch
        )
        
        # save checkpoints here
        if args.save_frequency > 0 and epoch % args.save_frequency == 0 or epoch==10: # early save at 10 and regular save checkpoints
            print(f"Saving model to {args.folder_save}/checkpoints/{epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_batch_idx': train_batch_idx,
                'loss': train_loss,
                'best_mse': best_mse,
            }, f"{args.folder_save}/checkpoints/{epoch}.pt")
            checkpoint_dir = f"{args.folder_save}/checkpoints"
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt') and int(f.split('.')[0]) < epoch:
                    os.remove(os.path.join(checkpoint_dir, f))

    writer.close()
        
def setup_model(args):
    """setup model, optimizer, (scheduler), loss_fn, start_epoch, train_batch_idx"""
    # find latest checkpoint    
    if args.should_resume:
        model_dir = f"{args.folder_save}/checkpoints"
        checkpoint_list = [int(f.split('.')[0]) for f in os.listdir(model_dir) if f.endswith('.pt')]
        if len(checkpoint_list) > 0: # checkpoint exists
            latest_model_idx = max(
                checkpoint_list
            )
            # print(f"Loading args from "+f"{args.folder_save}/args/args.pt")
            # args.__dict__.update(torch.load(f"{args.folder_save}/args/args.pt")['args'])
            args.path_to_load_model = f"{model_dir}/{latest_model_idx}.pt"
            args.checkpoint = {"epoch": latest_model_idx}
            args.should_resume = True
        else:
            args.path_to_load_model = ""
            args.should_resume = False      
    
    # initialize
    if args.task == 'MMNIST' or args.task == 'BBALL' or args.task == 'SPRITESMOT':
        model = BallModel(args).to(args.device)
    elif args.task == 'TRAFFIC4CAST':
        model = TrafficModel(args).to(args.device)
        raise NotImplementedError('traffic4cast not implemented')
    else:
        raise ValueError('not recognized task')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.01*args.lr, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2, eta_min=0.01*args.lr, last_epoch=- 1, verbose=True)
    start_epoch = 1
    train_batch_idx = 0
    best_mse = 1000.

    # load encoder+slot_attention
    if args.load_trained_slot_attention:
        print(f"load pretrained encoder and slot attention from {PRETRAINED_MODEL_PATH}")
        sa_autoae = SlotAttentionAutoEncoder(input_size=args.input_size, num_iterations=args.num_iterations_slot, num_slots=args.num_slots, slot_size=args.slot_size)
        sa_autoae.load_state_dict(torch.load(PRETRAINED_MODEL_PATH)['model_state_dict'])
        model.encoder.load_state_dict(sa_autoae.encoder.state_dict())
        model.slot_attention.load_state_dict(sa_autoae.slot_attention.state_dict())
        

    # resume model state dict
    if args.path_to_load_model != "":
        print('Resuming model from '+args.path_to_load_model)
        checkpoint = torch.load(args.path_to_load_model.strip(), map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming experiment id: {args.id}, from epoch: {start_epoch-1}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # self.__dict__.update(...) could cause unexpected probs
        train_batch_idx = checkpoint['train_batch_idx'] + 1 if 'train_batch_idx' in checkpoint else 0
        best_mse = checkpoint.get('best_mse', 1000.)
        print(f"Checkpoint resumed.")
        
    # setup loss_fn
    if args.loss_fn == "BCE":
        loss_fn = torch.nn.BCELoss() 
    elif args.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif args.loss_fn == 'MAE':
        loss_fn = torch.nn.L1Loss()
    else:
        loss_fn = torch.nn.MSELoss()

    return model, optimizer, scheduler, loss_fn, start_epoch, train_batch_idx, best_mse

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    main()

