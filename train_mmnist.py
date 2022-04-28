from tabnanny import check
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import wandb

from networks import BallModel, TrafficModel
from argument_parser import argument_parser
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from datasets import setup_dataloader
from tqdm import tqdm
from test_mmnist import dec_rim_util, test

import os 
from os import listdir
from os.path import isfile, join

print("Python Process PID: ", os.getpid())

set_seed(1997)

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
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # data: (labels, frames_in, frames_out)
        digit_labels, in_frames, out_frames = [tensor.to(args.device) for tensor in data] 
        data = torch.cat((in_frames, out_frames), dim=1) # [N, *T, 1, H, W]
        hidden = model.init_hidden(data.shape[0]).to(args.device)
        hidden = hidden.detach()
        memory = None
        if args.use_sw:
            memory = model.init_memory(data.shape[0]).to(args.device)
        
        optimizer.zero_grad()
        loss = 0.
        for frame in range(data.shape[1]-1):
            output, hidden, memory, intm = model(data[:, frame, :, :, :], hidden, memory)
            target = data[:, frame+1, :, :, :]
            loss += loss_fn(output, target)
            
        loss.backward()
        grad_norm = get_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False) 
        optimizer.step()
        writer.add_scalar('Grad Norm', grad_norm, train_batch_idx)

        train_batch_idx += 1 
        epoch_loss = epoch_loss + loss.detach()

    epoch_loss = epoch_loss / len(train_loader)

    return train_batch_idx, epoch_loss

def main():
    # parse and process args
    args = argument_parser()
    print(args)
    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")
    if not args.should_resume:
        make_dir(f"{args.folder_save}/checkpoints")
        make_dir(f"{args.folder_save}/args")
        print(f"Saving args to {args.folder_save}/args/args.pt")
        torch.save({
            "args": vars(args)
        }, f"{args.folder_save}/args/args.pt")

    # wandb setup
    project, name = args.experiment_name.split('_',1)
    wandb.init(project=project, name=name, config=vars(args), entity='nan-team')
    wandb_artf = wandb.Artifact(project+'_'+name+'_test'+str(wandb.run.id), type='predictions')
    columns = ['sample_id', 'frame_id', 'ground_truth', 'prediction', 'individual_prediction']
    if args.core == 'SCOFF':
        columns.append('rules_selected')

    # data setup
    train_loader, test_loader = setup_dataloader(args=args)

    # model setup
    model, optimizer, loss_fn, start_epoch, train_batch_idx = setup_model(args=args)
    
    # tensorboard setup
    writer = SummaryWriter(log_dir='./runs/'+args.id)

    # training loop
    for epoch in range(start_epoch, args.epochs+1):
        # train 
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        train_batch_idx, train_loss = train(
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
            "train loss": train_loss.item()
        }
        metric_dict = {
        }
        # scheduler.step(...)

        # test 
        if args.test_frequency > 0 and epoch % args.test_frequency == 0 or epoch <= 15:
            """test model accuracy and log intermediate variables here"""
            test_loss, prediction, data, metrics, test_table = test(
                model = model, 
                test_loader = test_loader, 
                args = args, 
                loss_fn = loss_fn, 
                writer = writer,
                rollout = True,
                epoch = epoch,
                log_columns=columns
            )
            loss_dict['test loss'] = test_loss
            test_mse = metrics['mse']
            test_f1 = metrics['f1']
            test_ssim = metrics['ssim']
            if args.core == 'RIM':
                rim_actv = metrics['rim_actv']
                rim_actv_mask = metrics['rim_actv_mask']
                dec_util = metrics['dec_util']
                most_used_units = metrics['most_used_units']
            elif args.core == 'SCOFF':
                rules_selected = metrics['rules_selected']
            blocked_dec = metrics['blocked_dec']
            # print out stats
            print(f"epoch {epoch}/{args.epochs} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | test mse: {test_mse:.4f} | "+\
            f"test F1 score: {test_f1:.4f} | test SSIM: {test_ssim:.4f}")

            # tensorboard

            writer.add_scalar(f'Metrics/MSE', test_mse, epoch)
            writer.add_scalar(f'Metrics/F1 Score', test_f1, epoch)
            writer.add_scalar(f'Metrics/SSIM', test_ssim, epoch)
            

            if args.core == 'RIM':
                writer.add_image('Stats/RIM Activation', rim_actv[0], epoch, dataformats='HW')
                writer.add_image('Stats/RIM Activation Mask', rim_actv_mask[0], epoch, dataformats='HW')
                writer.add_image('Stats/Unit Decoder Utilization', dec_util[0], epoch, dataformats='HW')
            
            if args.task == 'MMNIST':
                num_sample_to_record = 4
            elif args.task == 'BBALL':
                num_sample_to_record = 1
            elif args.task == 'TRAFFIC4CAST':
                num_sample_to_record = 1
            else:
                num_sample_to_record = 1
                print('Warning: unknown task type. ')
            cat_video = torch.cat(
                (data[0:num_sample_to_record, 1:, :, :, :], prediction[0:num_sample_to_record]),
                dim = 4 # join in width
            ) # N T C H W
            writer.add_video('Predicted Videos', cat_video, epoch)
            writer.add_video('Individual Predictions', blocked_dec[0], epoch) # N=num_blocks T 1 H W

            # wandb
            metric_dict = {
                'MSE': test_mse,
                'F1 Score': test_f1,
                'SSIM': test_ssim
            }
            stat_dict = {}
            if args.core == 'RIM':
                stat_dict.update({
                    'RIM Input Attention': wandb.Image(rim_actv[0].cpu()*256),
                    'RIM Activation Mask': wandb.Image(rim_actv_mask[0].cpu()*256),
                    'Unit Decoder Utilization': wandb.Image(dec_util[0].cpu()*256),
                    'Most Used Units in Decoder': wandb.Histogram(most_used_units), 
                })
            elif args.core == 'SCOFF':
                stat_dict.update({
                    'Rules Selected': wandb.Image(rules_selected[0].cpu()*256/9), # 0 to 9 classes
                })
            video_dict = {
                'Predicted Videos': wandb.Video(cat_video.cpu()*256, fps=3),
                'Individual Predictions': wandb.Video(blocked_dec[0].cpu()*256, fps=4),
            }
            wandb_artf.add(test_table, "predictions")
            wandb.run.log_artifact(wandb_artf)
            wandb.log({
                'Loss': loss_dict,
                'Metrics': metric_dict,
                'Stats': stat_dict,
                'Videos': video_dict,
            }, step=epoch)
        else:
            print(f"epoch {epoch}/{args.epochs} | "+\
                f"train loss: {train_loss:.4f}"
            )
            wandb.log({
                'Loss': loss_dict,
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
                'train_batch_idx': train_batch_idx,
                'loss': train_loss,
            }, f"{args.folder_save}/checkpoints/{epoch}.pt")

    writer.close()
        
def setup_model(args):
    """setup model, optimizer, (scheduler), loss_fn, start_epoch, train_batch_idx"""
    # find latest checkpoint    
    if args.should_resume:
        model_dir = f"{args.folder_save}/checkpoints"
        latest_model_idx = max(
            [int(f.split('.')[0]) for f in os.listdir(model_dir) if f.endswith('.pt')]
        )
        args.path_to_load_model = f"{model_dir}/{latest_model_idx}.pt"
        args.checkpoint = {"epoch": latest_model_idx}
    
    if args.path_to_load_model != "":
        print(f"Loading args from "+f"{args.folder_save}/args/args.pt")
        args.__dict__.update(torch.load(f"{args.folder_save}/args/args.pt"))
    
    # initialize
    if args.task == 'MMNIST':
        model = BallModel(args).to(args.device)
    elif args.task == 'BBALL':
        model = BallModel(args).to(args.device)
    elif args.task == 'TRAFFIC4CAST':
        model = TrafficModel(args).to(args.device)
        raise NotImplementedError('traffic4cast not implemented')
    else:
        raise ValueError('not recognized task')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # TODO later add scheduler here
    start_epoch = 1
    train_batch_idx = 0

    # resume model state dict
    if args.path_to_load_model != "":
        checkpoint = torch.load(args.path_to_load_model.strip())
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming experiment id: {args.id}, from epoch: {start_epoch-1}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # TODO later add scheduler here
        train_batch_idx = checkpoint['train_batch_idx'] + 1 if 'train_batch_idx' in checkpoint else 0
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

    return model, optimizer, loss_fn, start_epoch, train_batch_idx

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    main()


