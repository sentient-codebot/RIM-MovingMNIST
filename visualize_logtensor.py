'''
this script is intended to plot logged tensors.
entries in the logged tensor should be like:
    1-dim: tensor(idx) == scalar e.g. loss(scalar) vs. epoch(idx)
    2-dim: tensor(idx) == vector e.g. attn_scores(vector) vs. frame
'''
import torch
import matplotlib.pyplot as plt
from utils.visualize import plot_curve, plot_mat, HeatmapLog
from utils.util import make_dir
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Visualize Logged Tensor')
    parser.add_argument('--folder_log', type=str)
    args = parser.parse_args()

    return args

class TensorVisualizer:
    def __init__(self, folder_log, tensor_name):
        self.folder_log = folder_log
        self.tensor_name = tensor_name.replace(' ','_')
        self.save_folder = folder_log+'/'+self.tensor_name
        self.filename = folder_log+'/'+self.tensor_name+'/'+self.tensor_name
        self.tensor = None
        self.idx = None

    def log_tensor(self, epoch=None):
        if epoch is not None:
            self.idx, self.tensor = torch.load(self.filename+f'_epoch_{epoch}.pt')
        else:
            self.idx, self.tensor = torch.load(self.filename+f'.pt')
        return 0

    def plot_logged_tensor(self, epoch=None): # NOTE epoch for curve ????/
        save_path = self.save_folder
        make_dir(save_path)
        if epoch is not None:
            figname = self.tensor_name+f'_epoch_{epoch}.png'
        else:
            figname = self.tensor_name+f'.png'
        if self.tensor.dim() == 1:
            plot_curve(self.idx, self.tensor, save_path, figname)
        elif self.tensor.dim() == 2:
            # TODO plot a matrix as a Heatmap or?
            mat_log = HeatmapLog(save_path, 'figures')
            mat_log.plot(self.tensor, epoch)
        else:
            raise ValueError('tensor.dim should either be 1 or 2!')
        return 0

    def __call__(self, epoch=None):
        self.log_tensor(epoch)
        self.plot_logged_tensor(epoch)
        return 0


def main():
    pass
    # TODO parse a folder_log
    args = arg_parser()
    # TODO load the tensor
    loss_plot = TensorVisualizer(args.folder_log, "train_loss")
    testloss_plot =  TensorVisualizer(args.folder_log, "test_loss")
    rim_actv_plot = TensorVisualizer(args.folder_log, "rim_actv")
    decoder_actv = TensorVisualizer(args.folder_log, "decoder_actv")
    # gradnorm_plot = TensorVisualizer(args.folder_log, 'grad_norm')
    # testmat_plot = TensorVisualizer(args.folder_log, "test_mat")
    # TODO plot all-epoch tensors
    loss_plot()
    testloss_plot()
    rim_actv_plot()
    decoder_actv()

    # TODO plot per-epoch tensors
    for epoch_idx in [10,30,60,90,120]:
        # gradnorm_plot(epoch_idx)
        pass

if __name__ == "__main__":
    main()