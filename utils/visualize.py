import matplotlib.pyplot as plt
import torch
from .util import make_dir

def plot_frames(batch_of_pred, batch_of_target, start_frame, end_frame, batch_idx):
    '''
    batch_of_pred: (BATCH_SIZE, 50, W, H)
    batch_of_target: (BATCH_SIZE, 51, W, H) NOTICE: 51
    0 <= start_frame <= end_frame <= 50
    0 = data[:,1,:,:] 
    0 = pred[:,0,:,:]
    '''
    pred = batch_of_pred[batch_idx].detach().to(torch.device('cpu')).squeeze()
    target = batch_of_target[batch_idx].detach().to(torch.device('cpu')).squeeze()
    target = target[1:]
    num_frames = end_frame-start_frame+1
    fig, axs = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))
    for frame in range(start_frame, end_frame+1):
        axs[0, frame-start_frame].imshow(target[frame,:,:], cmap="Greys")
        axs[0, frame-start_frame].axis('off')
        axs[1, frame-start_frame].imshow(pred[frame,:,:], cmap="Greys")
        axs[1, frame-start_frame].axis('off')
    plt.savefig(f'frames_in_batch_{batch_idx}.png', dpi=120)
    plt.close()

def plot_curve(vector, save_path, filename):
    vector = vector.detach().to(torch.device('cpu')).squeeze()
    fig, axs = plt.subplots(1,1)
    axs.plot(vector)
    plt.savefig(save_path +'/'+ filename, dpi=120)
    plt.close()

def plot_mat(mat, mat_name, epoch):
    '''
    deprecated, use HeamapLog instead
    '''
    if mat.dim() == 3:
        mat_list = [mat[idx_unit,:,:].squeeze().cpu() for idx_unit in range(mat.shape[0])] 
    else:
        mat_list = [mat.cpu()]
    fig, axs = plt.subplots(1, len(mat_list), figsize=(2*len(mat_list), 2))
    for idx_mat, mat in enumerate(mat_list):
        if len(mat_list) == 1:
            axs.imshow(mat, cmap='hot', interpolation='nearest')
        else:
            axs[idx_mat].imshow(mat, cmap='hot', interpolation='nearest')
    fig.suptitle(mat_name.replace("_", " ")+ f' in epoch [{epoch}]')
    plt.savefig(mat_name + f'_epoch_{epoch}.png', dpi=120)
    plt.close()

class HeatmapLog:
    def __init__(self, folder_log, mat_name):
        mat_name = mat_name.replace(' ','_')
        make_dir(f"{folder_log}/"+mat_name)
        self.save_folder = f"{folder_log}/"+mat_name
        self.mat_name = mat_name

    def plot(self, mat, epoch=0):
        if mat.dim() == 3:
            mat_list = [mat[idx_unit,:,:].squeeze().cpu() for idx_unit in range(mat.shape[0])] 
        else:
            mat_list = [mat.cpu()]
        w_h_ratio = (mat_list[0].shape[1]/mat_list[0].shape[0])
        if w_h_ratio >= 1:
            fig, axs = plt.subplots(1, len(mat_list), figsize=(2*len(mat_list)*w_h_ratio, 2))
        else:
            fig, axs = plt.subplots(1, len(mat_list), figsize=(2*len(mat_list), 2/(w_h_ratio)))
        for idx_mat, mat in enumerate(mat_list):
            if len(mat_list) == 1:
                axs.imshow(mat, cmap='hot', interpolation='nearest')
            else:
                axs[idx_mat].imshow(mat, cmap='hot', interpolation='nearest')
        fig.suptitle(self.mat_name.replace("_", " ")+ f' in epoch [{epoch}]')
        plt.savefig(self.save_folder + '/' + self.mat_name + f'_epoch_{epoch}.png', dpi=120)
        plt.close()


class ScalarLog:
    def __init__(self, folder_log, var_name, epoch=None):
        self.save_folder = f"{folder_log}/"+var_name
        self.var_name = var_name
        make_dir(self.save_folder)
        self.var = []
        self.epoch = epoch

    def reset(self):
        self.var = []

    def append(self, value):
        self.var.append(value)

    def save(self):
        var_tensor = torch.tensor(self.var)
        if self.epoch is None:
            torch.save(var_tensor, self.save_folder +'/'+ self.var_name + '.pt')
        else:
            torch.save(var_tensor, self.save_folder +'/'+ self.var_name + f'_epoch_{epoch}.pt')

class VectorLog:
    def __init__(self, folder_log, var_name, epoch=None):
        '''
        log a vector in a list. vector is supposed to be 1-dim 
        '''
        self.save_folder = f"{folder_log}/"+var_name
        self.var_name = var_name
        make_dir(self.save_folder)
        self.var_stack = None
        self.epoch = epoch

    def reset(self):
        self.var_stack = None

    def append(self, vector):
        if self.var_stack is None:
            self.var_stack = vector.detach().unsqueeze(0)
        else:
            self.var_stack = torch.cat((self.var_stack, vector.detach().unsqueeze(0)), 0)

    def save(self):
        if self.epoch is None:
            torch.save(self.var_stack, self.save_folder +'/'+ self.var_name + '.pt')
        else:
            torch.save(self.var_stack, self.save_folder +'/'+ self.var_name + f'_epoch_{self.epoch}.pt')


def main():
    data = torch.rand((64,51,1,64,64))
    pred = torch.rand((64,50,1,64,64))
    error = torch.randn((100,1)) + torch.arange(100).unsqueeze(1)
    _t = torch.load('./../data.pt')
    _t = _t.unsqueeze(2)
    plot_frames(_t, _t, 0, 18, 6)
    # plot_curve(error)
    # epoch_losses = torch.load('../epoch_losses.pt')
    # plot_curve(epoch_losses)


if __name__ == "__main__":
    main()