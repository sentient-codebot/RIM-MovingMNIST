from time import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch 
import json
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from networks import BallModel
from argument_parser import argument_parser
from datasets import setup_dataloader
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir, _plot_mask
from statistics import mean
from utils.visualize import VecStack, make_grid_video, plot_heatmap, mplfig_to_video
from utils.logging import log_stats, enable_logging, setup_wandb_columns
from utils.metric import f1_score, gen_masks, get_mot_metrics, get_seg_mask, adjusted_rand_index
from utils.metric import consistency_measure
from tqdm import tqdm
import wandb
from utils import util

import utils.pssim.pytorch_ssim as pt_ssim

import os 
from os import listdir
from os.path import isfile, join

set_seed(1997)

MOT_TASKS = ['SPRITESMOT', 'VMDS', 'VOR']

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
def test(model, test_loader, args, loss_fn, writer, rollout=True, epoch=0, log_columns=None, calc_csty=False):
    '''test(model, test_loader, args, loss_fn, writer, rollout)'''
    start_time = time()
    # wandb table
    test_table = None
    if log_columns is not None:
        test_table = wandb.Table(columns=log_columns)

    if args.core == 'RIM':
        rim_actv = VecStack()
        rim_actv_mask = VecStack()
        dec_util = VecStack()
    if args.core == 'SCOFF':
        rule_attn_argmax = VecStack()
        rule_attn_probs_stack = VecStack()
        dec_util = VecStack()

    mse = lambda x, y: ((x - y)**2).mean(dim=(0,1,2)).sum() # x Shape: [batch_size, T, C, H, W]

    rollout_start = 10
    if args.task == 'MMNIST':
        rollout_start = 10
    elif args.task == 'BBALL':
        rollout_start = 20
    elif args.task in ['SPRITESMOT', 'VMDS', 'VOR']:
        rollout_start = 5
        rollout = False
        print("Rollout is turned off for task {}.".format(args.task))
    elif args.task == 'TRAFFIC4CAST':
        raise NotImplementedError('not set yet. ')


    model.eval()
    if args.task in MOT_TASKS:
        model.mot_eval = True

    epoch_loss = torch.tensor(0.).to(args.device)
    epoch_recon_loss = 0.
    epoch_pred_loss = 0.
    epoch_mseloss = torch.tensor(0.).to(args.device)
    f1 = 0.
    ssim = 0.
    most_used_units = []
    pred_list = []
    epoch_avr_len = 0.
    epoch_max_len = 0.
    epoch_ari = 0.
    id_counter = 0
    for batch_idx, data in enumerate(tqdm(test_loader, disable=not args.enable_tqdm)): # tqdm doesn't work here?
        if args.task == 'MMNIST':
            # data: (labels, frames_in, frames_out)
            digit_labels, in_frames, out_frames, obj_frames = [tensor.to(args.device) for tensor in data] 
            data = torch.cat((in_frames, out_frames), dim=1) # [N, *T, 1, H, W]
        elif args.task == 'MSPRITES':
            data, obj_frames = data # [N, T, C, H, W]
            data = data.to(args.device) # [N, K, T, C, H, W]
            obj_frames = obj_frames.to(args.device)
        else:
            data = data.to(args.device) # [N, T, c, H, W]
        hidden = model.init_hidden(data.shape[0]).to(args.device)
        memory = None
        if args.use_sw:
            memory = model.init_memory(data.shape[0]).to(args.device)
        if args.core == 'RIM':
            rim_actv.reset()
            rim_actv_mask.reset()
            dec_util.reset()
        if args.core == 'SCOFF':
            rule_attn_argmax.reset()
        data = data.to(args.device) # Shape: [batch_size, T, C, H, W] or [batch_size, T, H, W]
        if data.dim()==4:
            data = data.unsqueeze(2).float() # Shape: [batch_size, T, 1, H, W]
        hidden = hidden.detach()
        recon_loss = 0.
        pred_loss = 0.
        loss = 0.
        mseloss = 0.
        prediction = torch.zeros_like(data)
        slot_attn_probs = [] 
        slot_attn_map = []
        input_attn_probs = []
        rule_attn_probs_list = []
        rule_attn_probs_sm = []
        rule_attn_probs_gsm = []
        blocked_prediction = torch.zeros(
            (data.shape[0],
            args.num_hidden+1,
            data.shape[1],
            data.shape[2],
            data.shape[3],
            data.shape[4])
        ) # (BS, num_blocks, T, C, H, W)
        unmasked_ind_preds = blocked_prediction.clone()
        ind_pred = torch.empty((data.shape[0], args.num_hidden, data.shape[1]-rollout_start, data.shape[2], data.shape[3], data.shape[4]))
        reconstruction = []
        individual_recons = []
        soft_masks = [] # list of batches of masks
        ari_frame = []

        do_logging = batch_idx==len(test_loader)-1

        for frame in range(data.shape[1]-1):
            with torch.no_grad(), enable_logging(model, do_logging):
                if not rollout:
                    inputs = data[:, frame, :, :, :]
                elif frame >= rollout_start :
                    inputs = preds
                else:
                    inputs = data[:, frame, :, :, :]
                if args.task in ['SPRITESMOT', 'VMDS', 'VOR']:
                    if not args.spotlight_bias:
                        recons, preds, hidden, memory, object_mask = model(inputs, hidden, memory)
                    else:
                        recons, preds, hidden, memory, slot_means, slot_variances, attn_param_bias, object_mask = model(inputs, hidden, memory)
                    soft_masks.append(object_mask.squeeze(2)) # [BS, K, 1, H, W]
                else:
                    if not args.spotlight_bias:
                        recons, preds, hidden, memory = model(inputs, hidden, memory)
                    else:
                        recons, preds, hidden, memory, slot_means, slot_variances, attn_param_bias = model(inputs, hidden, memory)
                curr_target = inputs
                next_target = data[:, frame+1, :, :, :]
                if recons is not None:
                    recon_loss = recon_loss + loss_fn(recons, curr_target)
                pred_loss = pred_loss + loss_fn(preds, next_target)
                if args.spotlight_bias:
                    loss = loss + loss_fn(preds, next_target) + torch.sum(util.slot_loss(slot_means,slot_variances)) + 0.1*torch.sum(attn_param_bias**2)
                else:
                    loss = recon_loss + pred_loss
                
            # frame-wise metrics
            if 'SEP' in args.decoder_type and calc_csty:
                obj_pred_frame = get_seg_mask(model.hidden_features['individual_output'])
                obj_gt_frame = get_seg_mask(obj_frames[:,:,frame,...]) # ind_digits [N, K, T, C, H, W] -> [N, H*W, K]                
                ari_frame.append(adjusted_rand_index(obj_gt_frame, obj_pred_frame, reduction='mean').item())
            
            f1_frame = f1_score(next_target, preds)
            f1 += f1_frame

            prediction[:, frame+1, :, :, :] = preds
            if frame >= rollout_start:
                if 'individual_output' in model.hidden_features:
                    ind_pred[:, :, frame-rollout_start, :, :, :] = model.hidden_features['individual_output']
            if do_logging:
                blocked_prediction[:, 0, frame+1, :, :, :] = preds # dim == 6
                unmasked_ind_preds[:, 0, frame+1, :, :, :] = preds # dim == 6
                if 'individual_output' in model.hidden_features:
                    blocked_prediction[:, 1:, frame+1, :, :, :] = model.hidden_features['individual_output']
                    unmasked_ind_preds[:, 1:, frame+1, :, :, :] = model.hidden_features['individual_output_unmasked']
                if recons is not None:
                    reconstruction.append(recons) # [BS, C, H, W]
                    if not args.decode_hidden:
                        individual_recons.append(model.hidden_features['individual_recons']) # [BS, K, C, H, W]

                # wandb logging for table
                for sample_idx in range(data.shape[0]):
                    table_row = {
                        'sample_id': str(batch_idx)+'_'+str(sample_idx),
                        'frame_id': frame+1,
                        'prediction': wandb.Image(preds[sample_idx].detach().cpu()*255),
                        'ground_truth': wandb.Image(next_target[sample_idx].detach().cpu()*255),
                    }
                    if 'SEP' in args.decoder_type:
                        table_row['individual_prediction'] = wandb.Image(make_grid(model.hidden_features['individual_output'][sample_idx]*255, pad_value=255)) # N K C H W -> K C H W -> C *H **W
                        table_row['individual_prediction_unmasked'] = wandb.Image(make_grid(model.hidden_features['individual_output_unmasked'][sample_idx]*255, pad_value=255)) # N K C H W -> K C H W -> C *H **W
                    if args.core == 'RIM' or args.core == 'SCOFF':
                        table_row['input attention probs'] = wandb.Image(
                            plot_heatmap(
                                model.rnn_model.hidden_features['input_attention_probs'][sample_idx],  # [num_hidden, num_inputs]
                                x_label = 'Slots' if args.use_slot_attention else 'Features',
                                y_label = 'RIMs' if args.core == 'RIM' else 'OFs',
                                vmin=0.,
                                vmax=1.,
                                title=f'Frame {frame+1}', 
                            )
                        )
                    if args.core == 'SCOFF':
                        rule_attn_probs = model.rnn_model.hidden_features['rule_attn_probs'][sample_idx] # [num_hidden, num_rules]
                        for of_idx in range(args.num_hidden):
                            table_row.update({ 
                                f'rule_OF_{of_idx}': rule_attn_probs[of_idx].tolist(), # list of length num_rules, rule distribution
                            })
                    if log_columns is not None:
                        test_table.add_data(
                            *[table_row[col] for col in log_columns],
                        )
                        
                # wandb log
                if args.use_slot_attention:
                    slot_attn_probs.append(model.slot_attention.hidden_features['attention_probs']) # [batch_size, num_iter, num_slots, h, w]
                    slot_attn_map.append(model.slot_attention.hidden_features['attention_map']) # [batch_size, *, num_slots, h, w]
                if args.core == 'RIM' or args.core == 'SCOFF':
                # wandb/tb logging for concatenated image
                    dec_util.append(model.rnn_model.hidden_features.get("decoder_utilization", torch.zeros(1, 1)))
                most_used_units.append(0)
                if args.core == 'RIM':
                    rim_actv.append(model.rnn_model.hidden_features['input_attention_probs']) # shape (batchsize, num_units, 1) -> (BS, NU, T)
                    input_attn_probs.append(model.rnn_model.hidden_features['input_attention_probs'].unsqueeze(1)) # Shape: [N, 1, num_hidden, num_inputs]
                    rim_actv_mask.append(model.rnn_model.hidden_features["input_attention_mask"])
                    if 'rule_attn_probs' in model.rnn_model.hidden_features:
                        rule_attn_probs_list.append(model.rnn_model.hidden_features['rule_attn_probs'].unsqueeze(1)) # NOTE [N, 1, num_hidden, num_rules]
                elif args.core == 'SCOFF':
                    rule_attn_argmax.append(model.rnn_model.hidden_features['rule_attn_argmax']) # [N, num_hidden] -> [N, num_hidden, T]
                    rule_attn_probs_list.append(model.rnn_model.hidden_features['rule_attn_probs'].unsqueeze(1)) # NOTE [N, 1, num_hidden, num_rules]
                    if 'input_attention_probs' in model.rnn_model.hidden_features:
                        input_attn_probs.append(model.rnn_model.hidden_features['input_attention_probs'].unsqueeze(1)) # Shape: [N, 1, num_hidden, num_inputs]
                if 'rule_attn_probs_sm' in model.rnn_model.hidden_features:
                    rule_attn_probs_sm.append(model.rnn_model.hidden_features['rule_attn_probs_sm']) # [N, num_hidden, num_rules]
                if 'rule_attn_probs_gsm' in model.rnn_model.hidden_features:
                    rule_attn_probs_gsm.append(model.rnn_model.hidden_features['rule_attn_probs_gsm']) # [N, num_hidden, num_rules]
                    
        # calculate ARI across frames
        if len(ari_frame) > 0:
            epoch_ari += mean(ari_frame[-10:])
        
        # for MOT tasks, do one more step
        if args.task in ['SPRITESMOT', 'VMDS', 'VOR']:
            inputs = data[:, frame, :, :, :]
            with torch.no_grad():
                if not args.spotlight_bias:
                        recons, preds, hidden, memory, object_mask = model(inputs, hidden, memory)
                else:
                    recons, preds, hidden, memory, slot_means, slot_variances, attn_param_bias, object_mask = model(inputs, hidden, memory)
            soft_masks.append(object_mask.squeeze(2)) # [BS, K, H, W]
        
            pred_list = gen_masks(
                batch_size=data.shape[0],
                n_steps=len(soft_masks),
                n_slots=args.num_slots,
                id_counter=id_counter,
                pred_list=pred_list,
                soft_masks=torch.stack(soft_masks, dim=1).cpu(), # [BS, T, K, H, W]
            )
        
        # calculate consistency
        avr_len, max_len = None, None
        if 'SEP' in args.decoder_type and calc_csty:
            avr_len, max_len = consistency_measure(ind_pred, obj_frames[:, :, rollout_start:, ...], 
                                                   corr_padding=(1,1), output_ids=False, reduction='mean', exclude_background=True)
            epoch_avr_len += avr_len
            epoch_max_len += max_len    
        
        if not rollout:
            ssim += pt_ssim.ssim(data[:,1:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])), # data.shape = (batch, frame, 1, height, width)
                        prediction[:,1:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])))
            mseloss += mse(data[:,1:,:,:,:], prediction[:,1:,:,:,:]) # Shape: [N, T, C, H, W]
        else:
            ssim += pt_ssim.ssim(data[:,10:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])), # data.shape = (batch, frame, 1, height, width)
                        prediction[:,10:,:,:,:].reshape((-1,1,data.shape[3],data.shape[4])))
            mseloss += mse(data[:,10:,:,:,:], prediction[:,10:,:,:,:]) # Shape: [N, T, C, H, W]
            
        epoch_loss += loss.detach()
        epoch_recon_loss += recon_loss.detach() if isinstance(recon_loss, torch.Tensor) else recon_loss
        epoch_pred_loss += pred_loss.detach()
        epoch_mseloss += mseloss.detach()
        # if args.device == torch.device("cpu"):
        #     break

    prediction = prediction[:, 1:, :, :, :] # last batch of prediction, starting from frame 1
    blocked_prediction = blocked_prediction[:, :, 1:, :, :, :]
    unmasked_ind_preds = unmasked_ind_preds[:, :, 1:, :, :, :]
    if reconstruction:
        reconstruction = torch.stack(reconstruction, dim=1) # [N, T, C, H, W]
    if individual_recons:
        individual_recons = torch.stack(individual_recons, dim=2) # [N, K, T, C, H, W]
    object_masks = None
    if len(soft_masks) > 0:
        object_masks = torch.stack(soft_masks, dim=2).unsqueeze(3) # [BS, K, T, H, W]
    epoch_loss = epoch_loss / (batch_idx+1)
    epoch_recon_loss /= len(test_loader)
    epoch_pred_loss /= len(test_loader)
    epoch_mseloss = epoch_mseloss / (batch_idx+1)
    epoch_avr_len /= len(test_loader)
    epoch_max_len /= len(test_loader)
    epoch_ari /= len(test_loader)
    ssim = ssim / (batch_idx+1)
    f1_avg = f1 / (batch_idx+1) / (data.shape[1]-1)
    
    mot_metrics = None
    if args.task in ['SPRITESMOT', 'VMDS', 'VOR']:
        with open(args.mot_pred_file, 'w') as outfile:
            json.dump(pred_list, outfile) # args.folder_log+'/mot_json.json'
        mot_metrics = get_mot_metrics(args.mot_pred_file, args.mot_gt_file)

    if args.core == 'RIM':
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'rim_actv': rim_actv.show(),
            'input_attn_probs': torch.stack(input_attn_probs, dim=1), # Shape: [N, T, 1, num_hidden, num_inputs]
            'rim_actv_mask': rim_actv_mask.show(),
            'dec_util': dec_util.show(),
            'individual_output': blocked_prediction,
            'individual_output_unmasked': unmasked_ind_preds,
            'most_used_units': most_used_units
        }
        if 'rule_attn_probs' in model.rnn_model.hidden_features:
            metrics['rule_attn_probs'] = torch.stack(rule_attn_probs_list, dim=1) # Shape: [N, T, 1, num_hidden, num_rules]
    elif args.core == 'SCOFF':
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'individual_output': blocked_prediction,
            'individual_output_unmasked': unmasked_ind_preds,
            'input_attn_probs': torch.stack(input_attn_probs, dim=1), # Shape: [N, T, 1, num_hidden, num_inputs]
            'rule_attn_argmax': rule_attn_argmax.show(),
            'rule_attn_probs': torch.stack(rule_attn_probs_list, dim=1), # Shape: [N, T, 1, num_hidden, num_rules]
        }
    else:
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'individual_output': blocked_prediction,
            'individual_output_unmasked': unmasked_ind_preds,
        }
    if len(reconstruction) > 0:
        metrics['reconstruction'] = reconstruction
    if len(individual_recons) > 0:
        metrics['individual_recons'] = individual_recons
    if object_masks is not None:
        metrics['object_masks'] = object_masks
    if mot_metrics is not None:
        metrics['mot_metrics'] = mot_metrics
    if len(rule_attn_probs_sm) > 0:
        metrics['rule_attn_probs_sm'] = torch.stack(rule_attn_probs_sm, dim=1) # Shape: [N, T, num_hidden, num_rules]
    if len(rule_attn_probs_gsm) > 0:
        metrics['rule_attn_probs_gsm'] = torch.stack(rule_attn_probs_gsm, dim=1) # Shape: [N, T, num_hidden, num_rules]
    if 'SEP' in args.decoder_type and calc_csty:
        metrics['avr_len'] = epoch_avr_len
        metrics['max_len'] = epoch_max_len    
        metrics['ari'] = epoch_ari
        
    # slot attention
    if args.use_slot_attention:
        metrics['slot_attn_probs'] = torch.cat(slot_attn_probs, dim=1) # [batch_size, **, num_slots, h, w]
        metrics['slot_attn_map'] = torch.cat(slot_attn_map, dim=1) # [batch_size, **, num_slots, h, w]

    model.mot_eval = False
    print('test runtime:', time() - start_time)
    return epoch_loss, epoch_recon_loss, epoch_pred_loss, prediction, data, metrics, test_table


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
    # data setup
    train_loader, val_loader, test_loader = setup_dataloader(args=args)
    # resume args
    # print(f"Loading args from "+f"{args.folder_save}/args/args.pt")
    # args.__dict__.update(torch.load(f"{args.folder_save}/args/args.pt")['args'])
    if not args.should_resume:
        args.should_resume = True
    args.enable_tqdm = True
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
    
    # wandb setup
    project, name = args.id.split('_',1)
    wandb.init(project=project, name=name+'_test', config=vars(args), entity='nan-team', settings=wandb.Settings(start_method="thread"))
    print(args)
    columns = setup_wandb_columns(args)

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
    test_loss, recon_loss, pred_loss, prediction, data, metrics, test_table = test(
        model = model,
        test_loader = val_loader if args.use_val_set else test_loader,
        args = args,
        loss_fn = loss_fn,
        writer = writer,
        rollout = True,
        epoch = epoch,
        log_columns = columns,
        calc_csty = True if args.use_val_set or args.task == 'MSPRITES' else False,
    )
    log_stats(
        args=args,
        is_train=False,
        epoch=epoch,
        test_loss=test_loss,
        test_recon_loss=recon_loss,
        test_pred_loss=pred_loss,
        ground_truth=data,
        prediction=prediction,
        metrics=metrics,
        test_table=test_table,
        writer=writer,
        manual_init_scale=0. if not args.use_past_slots else torch.sigmoid(model.slot_attention.manual_init_scale_digit).detach()
    )

    writer.close()

    return None
        
def setup_model(args) -> torch.nn.Module:
    # initialize
    model = BallModel(args).to(args.device)
    
    # model options
    model.rnn_model.do_comm = args.do_comm
    
    if args.should_resume:
        if os.path.exists(f"{args.folder_save}/best_model/best.pt"):
            args.path_to_load_model = f"{args.folder_save}/best_model/best.pt"
            args.checkpoint = 'best'
        else:
            # Find the last checkpointed model and resume from that
            model_dir = f"{args.folder_save}/checkpoints"
            checkpoint_list = [int(f.split('.')[0]) for f in os.listdir(model_dir) if f.endswith('.pt')]
            if len(checkpoint_list) == 0:
                raise RuntimeError("No checkpoint found in {}".format(model_dir))
            latest_model_idx = max(
                checkpoint_list
            )
            args.path_to_load_model = f"{model_dir}/{latest_model_idx}.pt"
            args.checkpoint = {"epoch": latest_model_idx}

    if args.path_to_load_model != "":
        print(f"Resuming experiment id: {args.id} from {args.checkpoint}")
        checkpoint = torch.load(args.path_to_load_model.strip(), map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    return model, epoch

if __name__ == '__main__':
    main()


