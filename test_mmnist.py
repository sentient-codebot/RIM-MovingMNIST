from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from networks import BallModel
from argument_parser import argument_parser
from datasets import setup_dataloader
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import VecStack, make_grid_video
from utils.metric import f1_score
from tqdm import tqdm
import wandb

import utils.pssim.pytorch_ssim as pt_ssim

import os 
from os import listdir
from os.path import isfile, join

set_seed(1997)

# loss_fn = torch.nn.BCELoss()

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

# @torch.no_grad()
def test(model, test_loader, args, loss_fn, writer, rollout=True, epoch=0, log_columns=None):
    '''test(model, test_loader, args, loss_fn, writer, rollout)'''
    start_time = time()
    # wandb table
    if log_columns is not None:
        test_table = wandb.Table(columns=log_columns)

    previous_get_intm = model.get_intm
    model.get_intm = True
    if args.core == 'RIM':
        rim_actv = VecStack()
        rim_actv_mask = VecStack()
        dec_util = VecStack()
    if args.core == 'SCOFF':
        rules_selected = VecStack()

    mse = lambda x, y: ((x - y)**2).mean(dim=(0,1,2)).sum() # x Shape: [batch_size, T, C, H, W]

    if args.task == 'MMNIST':
        rollout_start = 10
    elif args.task == 'BBALL':
        rollout_start = 20
    elif args.task == 'TRAFFIC4CAST':
        raise NotImplementedError('not set yet. ')


    model.eval()

    epoch_loss = torch.tensor(0.).to(args.device)
    epoch_mseloss = torch.tensor(0.).to(args.device)
    f1 = 0.
    ssim = 0.
    most_used_units = []
    for batch_idx, data in enumerate(tqdm(test_loader) if __name__ == "__main__" else test_loader): # tqdm doesn't work here?
        if args.task == 'MMNIST':
            # data: (labels, frames_in, frames_out)
            digit_labels, in_frames, out_frames = [tensor.to(args.device) for tensor in data] 
            data = torch.cat((in_frames, out_frames), dim=1) # [N, *T, 1, H, W]
        else:
            data = data.to(args.device)
        hidden = model.init_hidden(data.shape[0]).to(args.device)
        memory = None
        if args.use_sw:
            memory = model.init_memory(data.shape[0]).to(args.device)
        if args.core == 'RIM':
            rim_actv.reset()
            rim_actv_mask.reset()
            dec_util.reset()
        if args.core == 'SCOFF':
            rules_selected.reset()
        data = data.to(args.device) # Shape: [batch_size, T, C, H, W] or [batch_size, T, H, W]
        if data.dim()==4:
            data = data.unsqueeze(2).float() # Shape: [batch_size, T, 1, H, W]
        hidden = hidden.detach()
        loss = 0.
        mseloss = 0.
        prediction = torch.zeros_like(data)
        blocked_prediction = torch.zeros(
            (data.shape[0],
            args.num_hidden+1,
            data.shape[1],
            data.shape[2],
            data.shape[3],
            data.shape[4])
        ) # (BS, num_blocks, T, C, H, W)

        for frame in range(data.shape[1]-1):
            with torch.no_grad():
                if not rollout:
                    output, hidden, memory, intm = model(data[:, frame, :, :, :], hidden, memory)
                elif frame >= rollout_start :
                    output, hidden, memory, intm = model(output, hidden, memory)
                else:
                    output, hidden, memory, intm = model(data[:, frame, :, :, :], hidden, memory)

                intm = intm._asdict()
                target = data[:, frame+1, :, :, :]
                prediction[:, frame+1, :, :, :] = output
                blocked_prediction[:, 0, frame+1, :, :, :] = output # dim == 6
                blocked_prediction[:, 1:, frame+1, :, :, :] = intm['blocked_dec']
                loss += loss_fn(output, target)

                f1_frame = f1_score(target, output)
                # writer.add_scalar(f'Metrics/F1 at Frame {frame}', f1_frame, epoch)
                f1 += f1_frame

                # wandb logging
                table_row = {
                    'sample_id': str(batch_idx)+'_'+'0',
                    'frame_id': frame+1,
                    'prediction': wandb.Image(output[0].detach().cpu()*255),
                    'ground_truth': wandb.Image(target[0].detach().cpu()*255),
                    'individual_prediction': wandb.Image(make_grid(intm['blocked_dec'][0]*255, pad_value=255)), # N K C H W -> K C H W -> C *H **W
                }
                if args.core == 'SCOFF':
                    rule_list = intm['rules_selected'][0].detach().cpu().tolist()
                    for of_idx in range(args.num_hidden):
                        table_row.update({
                            f'rule_OF_{of_idx}': rule_list[of_idx],
                        })
                if log_columns is not None:
                    test_table.add_data(
                        *[table_row[col] for col in log_columns],
                    )

            if __name__ == "__main__" and False:
                if not args.use_memory_for_decoder:
                    intm["decoder_utilization"] = dec_rim_util(model, hidden)
                else:
                    intm['decoder_utilization'] = dec_rim_util(model, memory)
                most_used_units.extend(torch.topk(intm["decoder_utilization"], k=args.num_hidden//2, dim=-1).indices.tolist())
            else:
                intm['decoder_utilization'] = torch.zeros(1, 1)
                most_used_units.append(0)
            
            if args.core == 'RIM':
                rim_actv.append(intm["input_attn"]) # shape (batchsize, num_units, 1) -> (BS, NU, T)
                rim_actv_mask.append(intm["input_attn_mask"])
                dec_util.append(intm["decoder_utilization"])
                pass
            elif args.core == 'SCOFF':
                rules_selected.append(intm["rules_selected"])
                pass
        
        if not rollout:
            ssim += pt_ssim.ssim(data[:,1:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])), # data.shape = (batch, frame, 1, height, width)
                        prediction[:,1:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])))
            mseloss += mse(data[:,1:,:,:,:], prediction[:,10:,:,:,:]) # Shape: [N, T, C, H, W]
        else:
            ssim += pt_ssim.ssim(data[:,10:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])), # data.shape = (batch, frame, 1, height, width)
                        prediction[:,10:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])))
            mseloss += mse(data[:,10:,:,:,:], prediction[:,10:,:,:,:]) # Shape: [N, T, C, H, W]
            
        epoch_loss += loss.detach()
        epoch_mseloss += mseloss.detach()
        if args.device == torch.device("cpu"):
            break
    

    prediction = prediction[:, 1:, :, :, :] # last batch of prediction, starting from frame 1
    blocked_prediction = blocked_prediction[:, :, 1:, :, :, :]
    epoch_loss = epoch_loss / (batch_idx+1)
    epoch_mseloss = epoch_mseloss / (batch_idx+1)
    ssim = ssim / (batch_idx+1)
    f1_avg = f1 / (batch_idx+1) / (data.shape[1]-1)

    if args.core == 'RIM':
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'rim_actv': rim_actv.show(),
            'rim_actv_mask': rim_actv_mask.show(),
            'dec_util': dec_util.show(),
            'blocked_dec': blocked_prediction,
            'most_used_units': most_used_units
        }
    elif args.core == 'SCOFF':
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'blocked_dec': blocked_prediction,
            'rules_selected': rules_selected.show(),
        }
    else:
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'blocked_dec': blocked_prediction
        }

    model.get_intm = previous_get_intm
    print('test runtime:', time() - start_time)
    return epoch_loss, prediction, data, metrics, test_table

@torch.no_grad()
def dec_rim_util(model, h):
    """check the contribution of the (num_module)-th RIM 
    
    Inputs:
        `model`: the model
        `h`: hidden state, [N, num_hidden, hidden_size]
    
    Outputs:
        `dec_util`: the decoder utilization, [N, num_hidden]
    """
    decoder_type = model.decoder_type
    # if decoder_type == 'CAT_BASIC':
    #   model.decoder(h) -> [N, 1, 64, 64]
    # elif decoder_type == 'SEP_SBD':
    #   model.deocder(h) -> fused, channels, alpha_mask
    
    if decoder_type == 'CAT_BASIC':
        func = lambda x: model.decoder(x.flatten(start_dim=1)).sum(dim=(1,2,3)) # [N]
    elif decoder_type == 'SEP_SBD':
        func = lambda x: model.decoder(x)[0].sum(dim=(1,2,3)) # [N]
    else:
        raise RuntimeError("Unknown decoder type")
    
    output_sum_grad = torch.autograd.functional.jacobian(func, h) # Shape: [N, N, num_hidden, hidden_size]
    output_sum_grad = torch.diagonal(output_sum_grad, dim1=0, dim2=1).movedim(-1, 0) # Shape: ... -> [num_hidden, hidden_size, N] -> [N, num_hidden, hidden_size]
    dec_util = output_sum_grad.abs().sum(dim=2) # Shape: [N, num_hidden]

    return dec_util


def main():
    # parse and process args
    args = argument_parser()
    print(f"Loading args from "+f"{args.folder_log}/args/args.pt")
    args.__dict__.update(torch.load(f"{args.folder_save}/args/args.pt"))
    if not args.should_resume:
        args.should_resume = True
    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    # wandb setup
    project, name = args.experiment_name.split('_',1)
    wandb.init(project=project, name=name+'_test', config=vars(args), entity='nan-team')
    print(args)
    wandb_artf = wandb.Artifact(project+'_'+name+'_test'+str(wandb.run.id), type='predictions')
    columns = ['sample_id', 'frame_id', 'ground_truth', 'prediction', 'individual_prediction']
    if args.core == 'SCOFF':
        for idx in range(args.num_hidden):
            columns.append('rule_OF_'+str(idx))

    # data setup
    train_loader, test_loader = setup_dataloader(args=args)

    # model setup
    model, epoch = setup_model(args=args)

    # TODO integrate to setup_model
    if args.loss_fn == "BCE":
        loss_fn = torch.nn.BCELoss() 
    elif args.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.MSELoss()    
    
    # tensorboard setup
    writer = SummaryWriter(log_dir='./runs/'+args.id+'_test')

    # call test function
    test_loss, prediction, data, metrics, test_table = test(
        model = model,
        test_loader = test_loader,
        args = args,
        loss_fn = loss_fn,
        writer = writer,
        rollout = True,
        epoch = epoch,
        log_columns = columns,
    )
    # unpack metrics
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
    blocked_dec = metrics['blocked_dec'] # dim == 6
    print(f"epoch [{epoch}] test loss: {test_loss:.4f}; test mse: {test_mse:.4f}; "+\
        f"test F1 score: {test_f1:.4f}; test SSIM: {test_ssim:.4f}")
    writer.add_scalar(f'Loss/Test Loss ({args.loss_fn.upper()})', test_loss, epoch)

    writer.add_scalar(f'Metrics/MSE', test_mse, epoch)
    writer.add_scalar(f'Metrics/F1 Score', test_f1, epoch)
    writer.add_scalar(f'Metrics/SSIM', test_ssim, epoch)

    if args.core == 'RIM':
        writer.add_image('Stats/RIM Activation', rim_actv[0], epoch, dataformats='HW')
        writer.add_image('Stats/RIM Activation Mask', rim_actv_mask[0], epoch, dataformats='HW')
        writer.add_image('Stats/Unit Decoder Utilization', dec_util[0], epoch, dataformats='HW')
    elif args.core == 'SCOFF':
        writer.add_image('Stats/Rules Selected', rules_selected[0], epoch, dataformats='HW')
    
    if args.task == 'MMNIST':
        num_sample_to_record = 4
    elif args.task == 'BBALL':
        num_sample_to_record = 1
    elif args.task == 'TRAFFIC4CAST':
        num_sample_to_record = 1
    else:
        num_sample_to_record = 1
        print('Warning: unknown task type. ')
    # concate video of ground truth and prediction
    cat_video = make_grid_video(data[0:num_sample_to_record, 1:, :, :, :],
                                prediction[0:num_sample_to_record], return_dim=5) 
    grided_ind_pred = make_grid_video(
        target = blocked_dec[0],
        return_dim = 5,
    )
    writer.add_video('Predicted Videos', cat_video, epoch)
    writer.add_video('Blocked Predictions', grided_ind_pred) # N num_blocks T 1 H W

    # wandb
    metric_dict = {
        'MSE': test_mse,
        'F1 Score': test_f1,
        'SSIM': test_ssim
    }
    stat_dict = {}
    if args.core == "RIM":
        stat_dict.update({
            'RIM Input Attention': wandb.Image(rim_actv[0].cpu()*255),
            'RIM Activation Mask': wandb.Image(rim_actv_mask[0].cpu()*255),
            'Unit Decoder Utilization': wandb.Image(dec_util[0].cpu()*255),
            'Most Used Units in Decoder': wandb.Histogram(most_used_units), # a list
        })
    elif args.core == 'SCOFF':
        stat_dict.update({
            'Rules Selected': wandb.Image(rules_selected[0].cpu()*255/9), # 0 to 9 classes
        })
    video_dict = {
        'Predicted Videos': wandb.Video((cat_video.cpu()*255).to(torch.uint8), fps=3),
        'Individual Predictions': wandb.Video((grided_ind_pred.cpu()*255).to(torch.uint8), fps=4),
    }
    wandb_artf.add(test_table, "predictions")
    wandb.run.log_artifact(wandb_artf)
    wandb.log({
        'Loss': {'test loss': test_loss},
        'Metrics': metric_dict,
        'Stats': stat_dict,
        'Videos': video_dict,
    }, step=epoch)

    writer.close()

    return None
        
def setup_model(args) -> torch.nn.Module:
    model = BallModel(args).to(args.device)
    
    if args.should_resume:
        # Find the last checkpointed model and resume from that
        model_dir = f"{args.folder_save}/checkpoints"
        # latest_model_idx = max(
        #     [int(model_idx) for model_idx in listdir(model_dir)
        #      if model_idx != "args"]
        # )
        latest_model_idx = max(
            [int(f.split('.')[0]) for f in os.listdir(model_dir) if f.endswith('.pt')]
        )
        args.path_to_load_model = f"{model_dir}/{latest_model_idx}.pt"
        args.checkpoint = {"epoch": latest_model_idx}

    if args.path_to_load_model != "":
        print(f"Loading args from "+f"{args.folder_save}/args/args.pt")
        args.__dict__.update(torch.load(f"{args.folder_save}/args/args.pt"))

        print(f"Resuming experiment id: {args.id} from epoch: {args.checkpoint}")
        checkpoint = torch.load(args.path_to_load_model.strip(), map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint else args.checkpoint['checkpoint']

    return model, epoch

if __name__ == '__main__':
    main()


