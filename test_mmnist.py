from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import autograd

from networks import BallModel
from argument_parser import argument_parser
from data.MovingMNIST import MovingMNIST
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from utils.visualize import ScalarLog, plot_frames, VectorLog, SaliencyMap, VecStack
from utils.metric import f1_score
from box import Box
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

    if args.core == 'RIM':
        rim_actv = VecStack()
    dec_actv = VecStack()

    mse = torch.nn.MSELoss()

    model.eval()

    epoch_loss = torch.tensor(0.).to(args.device)
    epoch_mseloss = torch.tensor(0.).to(args.device)
    f1 = 0.
    ssim = 0.
    for batch_idx, data in enumerate(test_loader): # tqdm doesn't work here?
        hidden = model.init_hidden(data.shape[0]).to(args.device)
        rim_actv.reset()
        dec_actv.reset()
        start_time = time()
        data = data.to(args.device)
        if data.dim()==4:
            data = data.unsqueeze(2).float()
        hidden = hidden.detach()
        loss = 0.
        mseloss = 0.
        prediction = torch.zeros_like(data)

        for frame in range(data.shape[1]-1):
            if frame == data.shape[1]-2: # last two frame
                hidden_before_last = hidden.detach()
            with torch.no_grad():
                if not rollout:
                    output, hidden, intm = model(data[:, frame, :, :, :], hidden)
                elif frame >= 5:
                    output, hidden, intm = model(output, hidden)
                else:
                    output, hidden, intm = model(data[:, frame, :, :, :], hidden)

                target = data[:, frame+1, :, :, :]
                prediction[:, frame+1, :, :, :] = output
                loss += loss_fn(output, target)
                mseloss += mse(output, target)
                f1_frame = f1_score(target, output)
                # writer.add_scalar(f'Metrics/F1 at Frame {frame}', f1_frame, epoch)
                if f1 is None:
                    f1 = f1_frame.reshape(-1, 1)

            intm["decoder_utilization"] = dec_rim_util(model, hidden, args)
            if args.core == 'RIM':
                rim_actv.append(intm["input_attn"][-1]) # shape (batchsize, num_units, 1)
            dec_actv.append(intm["decoder_utilization"][-1])
        
        ssim += pt_ssim.ssim(data[:,1:,:,:].reshape((-1,1,data.shape[3],data.shape[4])), # data.shape = (batch, frame, 1, height, width)
                        prediction[:,1:,:,:].reshape((-1,1,data.shape[3],data.shape[4])))
        epoch_loss += loss.detach()
        epoch_mseloss += mseloss.detach()
        if args.device == torch.device("cpu"):
            break
    
    prediction = prediction[:, 1:, :, :, :] # last batch of prediction, starting from frame 1
    epoch_loss = epoch_loss / (batch_idx+1)
    epoch_mseloss = epoch_mseloss / (batch_idx+1)
    ssim = ssim / (batch_idx+1)
    f1_avg = f1 / (batch_idx+1) / (data.shape[1]-1)

    metrics = {
        'mse': epoch_mseloss,
        'ssim': ssim,
        'f1': f1_avg,
        'rim_actv': rim_actv.show(),
        'dec_actv': dec_actv.show()
    }


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

    print(args)
    logbook = LogBook(config = args)

    if not args.should_resume:
        raise RuntimeError("args.should_resume should be set True. ")

    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    model = setup_model(args=args, logbook=logbook)

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
    
    test_loss, prediction, target, metrics= test(
        model = model,
        test_loader = test_loader,
        args = args,
        loss_fn = loss_fn,
        rollout = False,
        epoch = 0
    )
    f1_avg = metrics['f1']
    ssim = metrics['ssim']
    print(f"test loss: {test_loss}")
    print(f"test average F1 score: {f1_avg}")
    print(f'test SSIM: {ssim}')
    plot_frames(prediction, target, start_frame=1, end_frame=target.shape[1]-2, sample=[0,2,7,17,29,-1])

    # wait = input("Press any key to terminate program. ")
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
    
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id} from epoch: {args.checkpoint}")

    return model

if __name__ == '__main__':
    main()


