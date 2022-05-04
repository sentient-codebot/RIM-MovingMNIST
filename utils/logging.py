import wandb
from .visualize import make_grid_video
import torch

def log_stats(args, is_train, **kwargs):
    """
    log one sample of predictions in a batch
    """
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
    # metrics
    metrics = kwargs['metrics']
    mse = metrics.get('mse')
    f1 = metrics.get('f1')
    ssim = metrics.get('ssim')
    if args.core == 'RIM':
        input_attn_probs = metrics['input_attn_probs']
        rim_actv_mask = metrics['rim_actv_mask']
        dec_util = metrics['dec_util']
        most_used_units = metrics['most_used_units']
    elif args.core == 'SCOFF':
        input_attn_probs = metrics['input_attn_probs']
        rule_attn_argmax = metrics['rule_attn_argmax']
        rule_attn_probs = metrics.get('rule_attn_probs') 
    # videos patching
    individual_output = metrics['individual_output'] # dim == 6
    if args.task == 'MMNIST':
        num_vids = 4
    else:
        num_vids = 1
    grided_ind_pred = make_grid_video(
        target = individual_output[0],
        return_dim = 5,
    )
    cat_video = make_grid_video(ground_truth[0:num_vids, 1:, :, :, :],
                                prediction[0:num_vids], return_dim=5)

    # scalars
    #   tensorboard
    if writer is not None:
        writer.add_scalar('Learning Rate', lr, epoch)
        writer.add_scalar(f'Loss/Test Loss ({args.loss_fn.upper()})', test_loss, epoch)
        writer.add_scalar(f'Metrics/MSE', mse, epoch)
        writer.add_scalar(f'Metrics/F1 Score', mse, epoch)
        writer.add_scalar(f'Metrics/SSIM', mse, epoch)
    #   wandb
    loss_dict = {
        'test loss': test_loss,
    }
    if train_loss is not None:
        loss_dict['train loss'] = train_loss
    metric_dict = {
        'MSE': mse,
        'F1 Score': f1,
        'SSIM': ssim
    }
    stat_dict = {
        'Learning Rate': lr,
    }

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
            'Rules Selected': wandb.Image(rule_attn_argmax[0].cpu()*255/9), # 0 to 9 classes
        })

    # videos
    #   tensorboard
    if writer is not None:
        writer.add_video('Input Attention Probs', input_attn_probs[:1], epoch)
        if args.core == 'SCOFF':
            writer.add_video('Rule Attention Probs', rule_attn_probs[:1], epoch)
        writer.add_video('Predicted Videos', cat_video, epoch)
        writer.add_video('Individual Predictions', grided_ind_pred) # N num_blocks T 1 H W
    #   wandb
    video_dict = {
        'Input Attention Probs': wandb.Video((input_attn_probs[:1].cpu()*255).to(torch.uint8)), # [1, T, 1, H, W]
        'Predicted Videos': wandb.Video((cat_video.cpu()*255).to(torch.uint8), fps=3),
        'Individual Predictions': wandb.Video((grided_ind_pred.cpu()*255).to(torch.uint8), fps=4),
    }
    if args.core == 'SCOFF':
        video_dict.update({
            'Rule Attention Probs': wandb.Video((rule_attn_probs[:1].cpu()*255).to(torch.uint8)), # [1, T, 1, H, W]
        })
    
    # histograms
    if args.core == 'RIM':
        stat_dict.update({
            'Most Used Units in Decoder': wandb.Histogram(most_used_units), # a list
        })

    # wandb log
    project, name = args.id.split('_',1)
    if is_train:
        wandb_artf = wandb.Artifact(project+'_'+name, type='predictions', metadata=vars(args).update({'epoch': epoch}))
    else:
        wandb_artf = wandb.Artifact(project+'_'+name+'_test', type='predictions', metadata=vars(args).update({'epoch': epoch}))
    wandb_artf.add(test_table, "predictions")
    wandb.run.log_artifact(wandb_artf)
    wandb.log({
        'Loss': loss_dict,
        'Metrics': metric_dict,
        'Stats': stat_dict,
        'Videos': video_dict,
    }, step=epoch)

    # print 
    if is_train:
        print(f"epoch {epoch}/{args.epochs} | "+\
            f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | "+\
            f"test mse: {mse:.4f} | "+\
            f"test F1 score: {f1:.4f} | test SSIM: {ssim:.4f}")
    else:
         print(f"epoch {epoch}/{args.epochs} | "+\
            f"test loss: {test_loss:.4f} | "+\
            f"test mse: {mse:.4f} | "+\
            f"test F1 score: {f1:.4f} | test SSIM: {ssim:.4f}")

class enable_logging():
    """enable logging of a nn.Module by setting 'do_logging' to True/False 
    
    `with enable_logging(model, do_logging): `

    """
    def __init__(self, model: torch.nn.Module, do_logging: bool = True):
        self.model = model
        self.prev = getattr(self.model, 'do_logging', False)
        self.do_logging = do_logging

    def __enter__(self,):
        self.model.do_logging = self.do_logging

    def __exit__(self, type, value, traceback):
        self.model.do_logging = self.prev
