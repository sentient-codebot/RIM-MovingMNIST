from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd
from torch.utils.tensorboard import SummaryWriter

from networks import BallModel
from argument_parser import argument_parser
from data.MovingMNIST import MovingMNIST
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import ScalarLog, plot_frames, VectorLog, SaliencyMap, VecStack
from utils.metric import f1_score
from tqdm import tqdm

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
def test(model, test_loader, args, loss_fn, writer, rollout=True, epoch=0):
    '''test(model, test_loader, args, loss_fn, writer, rollout)'''
    previous_get_intm = model.get_intm
    model.get_intm = True
    if args.core == 'RIM':
        rim_actv = VecStack()
        rim_actv_mask = VecStack()
    # dec_actv = VecStack()

    mse = lambda x, y: ((x - y)**2).mean(dim=(0,1,2)).sum() # x Shape: [batch_size, T, C, H, W]

    model.eval()

    epoch_loss = torch.tensor(0.).to(args.device)
    epoch_mseloss = torch.tensor(0.).to(args.device)
    f1 = 0.
    ssim = 0.
    for batch_idx, data in enumerate(test_loader): # tqdm doesn't work here?
        hidden = model.init_hidden(data.shape[0]).to(args.device)
        if args.core == 'RIM':
            rim_actv.reset()
            rim_actv_mask.reset()
        # dec_actv.reset()
        start_time = time()
        data = data.to(args.device)
        if data.dim()==4:
            data = data.unsqueeze(2).float()
        hidden = hidden.detach()
        loss = 0.
        mseloss = 0.
        prediction = torch.zeros_like(data)
        blocked_prediction = torch.zeros(
            (data.shape[0],
            args.num_units+1,
            data.shape[1],
            data.shape[2],
            data.shape[3],
            data.shape[4])
        ) # (BS, num_blocks, T, C, H, W)

        for frame in range(data.shape[1]-1):
            if frame == data.shape[1]-2: # last two frame
                hidden_before_last = hidden.detach()
            with torch.no_grad():
                if not rollout:
                    output, hidden, reg_loss, intm = model(data[:, frame, :, :, :], hidden)
                elif frame >= 10:
                    output, hidden, reg_loss, intm = model(output, hidden)
                else:
                    output, hidden, reg_loss, intm = model(data[:, frame, :, :, :], hidden)

                intm = intm._asdict()
                target = data[:, frame+1, :, :, :]
                prediction[:, frame+1, :, :, :] = output
                blocked_prediction[:, 0, frame+1, :, :, :] = output
                blocked_prediction[:, 1:, frame+1, :, :, :] = intm['blocked_dec']
                loss += loss_fn(output, target)

                f1_frame = f1_score(target, output)
                # writer.add_scalar(f'Metrics/F1 at Frame {frame}', f1_frame, epoch)
                f1 += f1_frame

            # intm["decoder_utilization"] = dec_rim_util(model, hidden, args)
            if args.core == 'RIM':
                rim_actv.append(intm["input_attn"]) # shape (batchsize, num_units, 1) -> (BS, NU, T)
                rim_actv_mask.append(intm["input_attn_mask"])
            # dec_actv.append(intm["decoder_utilization"])
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
            # 'dec_actv': dec_actv.show(),
            'blocked_dec': blocked_prediction
        }
    else:
        metrics = {
            'mse': epoch_mseloss,
            'ssim': ssim,
            'f1': f1_avg,
            'blocked_dec': blocked_prediction
        }

    model.get_intm = previous_get_intm
    return epoch_loss, prediction, data, metrics

def dec_rim_util(model, h, args):
    """check the contribution of the (num_module)-th RIM by seeing how much they contribute to the activaiton of first relu"""
    h = h.clone().detach().requires_grad_(True)
    
    h_flat = h.view(h.shape[0],-1)
    decoded = model.Decoder(h_flat)
    grad = torch.autograd.grad(decoded.sum(), h)[0]

    util_dec = torch.sum(torch.abs(grad), dim=2)
    return util_dec


def main():
    args = argument_parser()
    print(f"Loading args from "+f"{args.folder_log}/model/args")
    args.__dict__.update(torch.load(f"{args.folder_log}/model/args"))

    print(args)
    logbook = LogBook(config = args)

    if not args.should_resume:
        args.should_resume = True

    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    model, epoch = setup_model(args=args, logbook=logbook)

    args.directory = './data' # dataset directory
    test_set = MovingMNIST(root='./data', train=False, download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True
    )

    if args.loss_fn == "BCE":
        loss_fn = torch.nn.BCELoss() 
    elif args.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.MSELoss()    
    
    writer = SummaryWriter(log_dir='./runs/'+args.id+'_test')
    test_loss, prediction, data, metrics= test(
        model = model,
        test_loader = test_loader,
        args = args,
        loss_fn = loss_fn,
        writer = writer,
        rollout = True,
        epoch = epoch
    )
    test_mse = metrics['mse']
    test_f1 = metrics['f1']
    test_ssim = metrics['ssim']
    rim_actv = metrics['rim_actv']
    rim_actv_mask = metrics['rim_actv_mask']
    # dec_actv = metrics['dec_actv']
    blocked_dec = metrics['blocked_dec']
    print(f"epoch [{epoch}] test loss: {test_loss:.4f}; test mse: {test_mse:.4f}; "+\
        f"test F1 score: {test_f1:.4f}; test SSIM: {test_ssim:.4f}")
    writer.add_scalar(f'Loss/Test Loss ({args.loss_fn.upper()})', test_loss, epoch)

    writer.add_scalar(f'Metrics/MSE', test_mse, epoch)
    writer.add_scalar(f'Metrics/F1 Score', test_f1, epoch)
    writer.add_scalar(f'Metrics/SSIM', test_ssim, epoch)

    writer.add_image('Stats/RIM Activation', rim_actv[0], epoch, dataformats='HW')
    writer.add_image('Stats/RIM Activation Mask', rim_actv_mask[0], epoch, dataformats='HW')
    # writer.add_image('Stats/RIM Decoder Utilization', dec_actv[0], epoch, dataformats='HW')
    cat_video = torch.cat(
        (data[0:4, 1:, :, :, :],prediction[0:4]),
        dim = 3 # join in height
    ) # N T C H W
    writer.add_video('Predicted Videos', cat_video, epoch)
    writer.add_video('Blocked Predictions', blocked_dec[0]) # N=num_blocks T 1 H W

    hidden = model.init_hidden(data.shape[0]).to(args.device)
    # writer.add_graph(model, (data[:, 0, :, :, :], hidden))
    # plot_frames(prediction, data, start_frame=1, end_frame=data.shape[1]-2, sample=[0,2,7,17,29,-1])

    # wait = input("Press any key to terminate program. ")
    writer.close()

    return None
        
def setup_model(args, logbook) -> torch.nn.Module:
    model = BallModel(args)
    
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
        checkpoint = torch.load(args.path_to_load_model.strip(), map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
    
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id} from epoch: {args.checkpoint}")

    return model, epoch

if __name__ == '__main__':
    main()


