from matplotlib.pyplot import xlabel
from argparse import Namespace
from time import time
import wandb
from .visualize import make_grid_video, heatmap_to_video, plot_heatmap
import torch
import os
from functools import partial

def log_stats(args, is_train, **kwargs):
    """
    log one sample of predictions in a batch
    """
    start_time = time()
    # 
    epoch = kwargs.get('epoch', 0)
    lr = kwargs.get('lr', None)
    ground_truth = kwargs.get('ground_truth', None)
    prediction = kwargs.get('prediction', None)
    test_table = kwargs.get('test_table', None)
    # tensorboard
    writer = kwargs.get('writer', None)
    # loss
    test_loss = kwargs.get('test_loss')
    train_loss = kwargs.get('train_loss')
    train_recon_loss = kwargs.get('train_recon_loss')
    train_pred_loss = kwargs.get('train_pred_loss')
    test_recon_loss = kwargs.get('test_recon_loss')
    test_pred_loss = kwargs.get('test_pred_loss')
    # model parameters
    manual_init_scale = kwargs.get('manual_init_scale', None)
    # metrics
    metrics = kwargs['metrics']
    mse = metrics.get('mse')
    f1 = metrics.get('f1')
    ssim = metrics.get('ssim')
    if args.core == 'RIM':
        rim_actv_mask = metrics['rim_actv_mask']
        dec_util = metrics['dec_util']
        most_used_units = metrics['most_used_units']
        if args.use_rule_sharing:
            rule_attn_probs = metrics['rule_attn_probs']
    elif args.core == 'SCOFF':
        rule_attn_argmax = metrics['rule_attn_argmax'] # LongTensor
        rule_attn_probs = metrics.get('rule_attn_probs')
    input_attn_probs = metrics.get('input_attn_probs')
    rule_attn_probs_sm = metrics.get('rule_attn_probs_sm')
    rule_attn_probs_gsm = metrics.get('rule_attn_probs_gsm')
    reconstruction = metrics.get('reconstruction') # Tensor|None
    individual_recons = metrics.get('individual_recons') # Tensor|None
    object_masks = metrics.get('object_masks') # Tensor|None
    slot_attn_probs = metrics.get('slot_attn_probs') # Tensor|None, [batch_size, num_iter*num_frames, num_slots, h, w]
    slot_attn_map = metrics.get('slot_attn_map') # Tensor|None
    avr_len = metrics.get('avr_len')
    max_len = metrics.get('max_len')
    ari = metrics.get('ari')
    # videos patching
    individual_output = metrics['individual_output'] # dim == 6
    ind_pred_unmasked = metrics['individual_output_unmasked'] # dim == 6
    if args.task == 'MMNIST':
        num_vids = 4
    else:
        num_vids = 1
    grided_ind_pred = (make_grid_video(
        target = individual_output[0],
        return_dim = 5,
    )*255).to(torch.uint8).cpu()
    grided_ind_pred_unmasked = (make_grid_video(
        target = ind_pred_unmasked[0],
        return_dim = 5,
    )*255).to(torch.uint8).cpu()
    gt_preds_video = (make_grid_video(ground_truth[0:num_vids, 1:, :, :, :],
                                prediction[0:num_vids], return_dim=5)*255).to(torch.uint8).cpu()
    gt_recons_video = None
    if reconstruction is not None:
        # [N, T, C, H, W]
        frame_length = reconstruction.shape[1]
        gt_recons_video = (
            make_grid_video(
                ground_truth[0:num_vids, 0:frame_length, :, :, :],
                reconstruction[0:num_vids, 0:frame_length, :, :, :],
                return_dim=5
            )*255
        ).to(torch.uint8).cpu()
    grided_ind_recon = None
    if individual_recons is not None:
        grided_ind_recon = (
            make_grid_video(
                individual_recons[0],
                return_dim=5
            )*255
        ).to(torch.uint8).cpu()
    grided_obj_masks = None
    if object_masks is not None:
        grided_obj_masks = (
            make_grid_video(
                object_masks[0],
                return_dim=5
            )*255
        ).to(torch.uint8).cpu()
    # slot attention
    if slot_attn_probs is not None:
        # [batch_size, num_iter*num_frames, num_slots, h, w]
        titles = [f'Slot {i}' for i in range(slot_attn_probs.shape[2])]
        slot_attn_probs_vid = heatmap_to_video(
                    slot_attn_probs[0], # [num_iter*num_frames, num_slots, h, w]
                    title=titles,
                    cbar=True,
                    linewidth=0.,
                    cmap='OrRd',
                )
    if slot_attn_map is not None:
        # [batch_size, num_iter*num_frames, num_slots, h, w]
        slot_attn_map = slot_attn_map - slot_attn_map.min()
        slot_attn_map = slot_attn_map / (slot_attn_map.max() + 1e-8) # [0, 1]
        titles = [f'Slot {i}' for i in range(slot_attn_probs.shape[2])]
        slot_attn_map_vid = heatmap_to_video(
                    slot_attn_map[0], # [num_iter*num_frames, num_slots, h, w]
                    title=titles,
                    cbar=False,
                    linewidth=0.,
                    cmap='OrRd',
                )

    # scalars
    #   tensorboard
    if writer is not None:
        writer.add_scalar(f'Loss/Test Loss ({args.loss_fn.upper()})', test_loss, epoch)
        writer.add_scalar(f'Metrics/MSE', mse, epoch)
        writer.add_scalar(f'Metrics/F1 Score', mse, epoch)
        writer.add_scalar(f'Metrics/SSIM', mse, epoch)
        if is_train:
             writer.add_scalar('Learning Rate', lr, epoch)
    #   wandb
    loss_dict = {
        'test loss': test_loss,
        'test recon loss': test_recon_loss,
        'test pred loss': test_pred_loss,
    }
    if train_loss is not None:
        loss_dict.update({
            'train loss': train_loss,
            'train recon loss': train_recon_loss,
            'train pred loss': train_pred_loss
        })
    metric_dict = {
        'MSE': mse,
        'F1 Score': f1,
        'SSIM': ssim
    }
    if 'mot_metrics' in metrics:
        metric_dict.update(metrics['mot_metrics'])
    stat_dict = {
        'Learning Rate': lr,
    }
    if avr_len is not None and max_len is not None:
        stat_dict.update(
            {
                'Average Consistent Length': avr_len,
                'Maximum Consistent Length': max_len,
            }
        )
    if ari is not None:
        stat_dict['ARI'] = ari
    if manual_init_scale is not None:
        stat_dict['Past Slot Init Scale'] = manual_init_scale

    # images
    #   tensorboard
    if writer is not None:
        if args.core == 'RIM':
            writer.add_image('Stats/RIM Activation Mask', rim_actv_mask[0], epoch, dataformats='HW')
            writer.add_image('Stats/Unit Decoder Utilization', dec_util[0], epoch, dataformats='HW')
        elif args.core == 'SCOFF':
            writer.add_image('Stats/Most Likely Rule', rule_attn_argmax[0], epoch, dataformats='HW')
    #   wandb
    if args.core == "RIM":
        stat_dict.update({
            # 'RIM Input Attention Probs': wandb.Image(input_attn_probs[0].cpu()*255),
            'RIM Activation Mask': wandb.Image(rim_actv_mask[0].cpu()*255),
            'Unit Decoder Utilization': wandb.Image(dec_util[0].cpu()*255),
        })
    elif args.core == 'SCOFF':
        stat_dict.update({
            # 'SCOFF Input Attention Probs': wandb.Image(input_attn_probs[0].cpu()*255),
            'Rules Selected': wandb.Image(
                plot_heatmap(
                    rule_attn_argmax[0], # LongTensor, full shape [N, num_hidden, T]
                    x_label='Frames',
                    y_label='OFs',
                    vmin=0,
                    vmax=args.num_rules-1,
                    fmt='d',
                    annot=True,
                ),
                rule_attn_argmax[0].cpu()*255/9), # 0 to 9 classes
        })

    # videos
    #   tensorboard
    if writer is not None:
        if input_attn_probs is not None:
                writer.add_video('Input Attention Probs', input_attn_probs[:1], epoch)
        if args.core == 'SCOFF' or args.use_rule_sharing:
            writer.add_video('Rule Attention Probs', rule_attn_probs[:1], epoch)
        writer.add_video('Predicted Videos', gt_preds_video, epoch)
        writer.add_video('Individual Predictions', grided_ind_pred) # N num_blocks T 1 H W
        writer.add_video('Individual Predictions (unmasked)', grided_ind_pred_unmasked) # N num_blocks T 1 H W
    #   wandb
    video_dict = {
        'Predicted Videos': wandb.Video(gt_preds_video, fps=3),
        
    }
    if 'SEP' in args.decoder_type:
        video_dict.update(
            {
                'Individual Predictions': wandb.Video(grided_ind_pred, fps=3),
                'Individual Predictions (unmasked)': wandb.Video(grided_ind_pred_unmasked, fps=3),
            }
        )
    if input_attn_probs is not None:
        video_dict.update({
            # 'Input Attention Probs': wandb.Video((input_attn_probs[:1].cpu()*255).to(torch.uint8)), # [1, T, 1, H, W]
            'Input Attention Probs': wandb.Video(
                heatmap_to_video(
                    input_attn_probs[0], 
                    x_label='Slots' if args.use_slot_attention else 'Features',
                    y_label='RIMs' if args.core=='RIM' else 'OFs',
                    vmin=0.,
                    vmax=1.,
                ),
                fps=1,
            ), # [T, 3, H, W]
        })
    if rule_attn_probs_sm is not None:
        video_dict['Rule Attention Probs Softmax'] = wandb.Video(
            heatmap_to_video(
                    rule_attn_probs_sm[0], 
                    x_label='Rules',
                    y_label='OFs',
                    vmin=0.,
                    vmax=1.,
                ),
            fps=1,
        )
    if rule_attn_probs_gsm is not None:
        video_dict['Rule Attention Probs Gumbel-Softmax'] = wandb.Video(
            heatmap_to_video(
                    rule_attn_probs_gsm[0], 
                    x_label='Rules',
                    y_label='OFs',
                    vmin=0.,
                    vmax=1.,
                ),
            fps=1,
        )
    if args.core == 'SCOFF' or args.use_rule_sharing:
        video_dict.update({
            'Rule Attention Probs': wandb.Video(
                heatmap_to_video(
                    rule_attn_probs[0],
                    x_label='Rules/Schemata',
                    y_label='OFs',
                    vmin=0.,
                    vmax=1.,
                ),
                fps=1,
            ), # [T, 3, H, W]
        })
    if gt_recons_video is not None:
        video_dict['Reconstructed Videos'] = wandb.Video(gt_recons_video, fps=3)
    if grided_ind_recon is not None:
        video_dict['Individual Reconstructions'] = wandb.Video(grided_ind_recon, fps=3)
    if grided_obj_masks is not None:
        video_dict['Object Masks'] = wandb.Video(grided_obj_masks, fps=3)
    if slot_attn_probs is not None:
        video_dict['Slot Attention Probs'] = wandb.Video(
            slot_attn_probs_vid, fps=1,
        )
    if slot_attn_map is not None:
        video_dict['Slot Attention Map'] = wandb.Video(
            slot_attn_probs_vid, fps=1,
        )
    
    # histograms
    if args.core == 'RIM':
        stat_dict.update({
            'Most Used Units in Decoder': wandb.Histogram(most_used_units), # a list
        })

    # wandb log
    project, name = args.id.split('_',1)
    # - artifact
    if test_table is not None:
        if not os.environ.get('DISABLE_ARTIFACT', False):
            try:
                metadata = vars(args)
                metadata['epoch'] = epoch
                if is_train:
                    wandb_artf = wandb.Artifact(project+'_'+name, type='predictions', metadata=metadata)
                    wandb_artf.add(test_table, "predictions")
                else:
                    wandb_artf = wandb.Artifact(project+'_'+name, type='predictions', metadata=metadata)
                    wandb_artf.add(test_table, "predictions")
                print('logging artifact')
                wandb.run.log_artifact(wandb_artf)
            except OSError as e:
                print('OSError occurred:', e)
    # - log
    wandb.log({
        'Loss': loss_dict,
        'Metrics': metric_dict,
        'Stats': stat_dict,
        'Videos': video_dict,
        'Epoch': epoch,
    })
    runtime = time() - start_time
    print('logging runtime:', runtime)

    # print 
    if is_train:
        message = f"epoch {epoch}/{args.epochs} | "+\
            f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | "+\
            f"test mse: {mse:.4f} | "+\
            f"test F1 score: {f1:.4f} | test SSIM: {ssim:.4f}"
    else:
        message = f"epoch {epoch}/{args.epochs} | "+\
            f"test loss: {test_loss:.4f} | "+\
            f"test mse: {mse:.4f} | "+\
            f"test F1 score: {f1:.4f} | test SSIM: {ssim:.4f}"
    if avr_len is not None:
        message += f" | test acl {avr_len:.4f} | test mcl {max_len:.4f}"
    if ari is not None:
        message += f" | test ari {ari:.4f}"
    print(message)

class enable_logging():
    """enable logging of a nn.Module by setting 'do_logging' to True/False 
    
    `with enable_logging(model, do_logging): `

    """
    def __init__(self, model: torch.nn.Module, do_logging: bool = True):
        self.model = model
        self.prev = getattr(self.model, 'do_logging', False)
        self.do_logging = do_logging
        def set_logging_(module, do_logging):
            module.do_logging = do_logging
        
        self.set_logging = partial(set_logging_, do_logging=self.do_logging)
        self.reset_logging = partial(set_logging_, do_logging=self.prev)

    def __enter__(self,):
        self.model.apply(self.set_logging)

    def __exit__(self, type, value, traceback):
        self.model.apply(self.reset_logging)

def setup_wandb_columns(args: Namespace) -> list[str]:
    """
    setup wandb artifact columns"""
    columns = ['sample_id', 'frame_id', 'ground_truth', 'prediction',]
    if 'SEP' in args.decoder_type:
        columns.append('individual_prediction')
        columns.append('individual_prediction_unmasked')
    if args.core =='RIM' or args.core == 'SCOFF':
        columns.append('input attention probs')
    if args.core == 'SCOFF':
        for idx in range(args.num_hidden):
            columns.append('rule_OF_'+str(idx))
            
    return columns
