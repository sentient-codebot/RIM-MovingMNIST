from multiprocessing.dummy import Array
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import seaborn as sns
from torch import Tensor
from torchvision.utils import make_grid
from wandb import Video
from .util import make_dir
import argparse
from typing import List, Sequence, Union, Any, Optional
from numpy import ndarray
import numpy as np
import wandb
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

ArrayLike = Union[ndarray, Tensor, list[list]]
FigSize = Union[
    int,
    float,
    tuple[int],
    tuple[float],
]

def plot_frames(batch_of_pred, batch_of_target, start_frame, end_frame, sample):
    '''
    batch_of_pred: (BATCH_SIZE, 51, W, H)
    batch_of_target: (BATCH_SIZE, 51, W, H) NOTICE: 51
    0 <= start_frame <= end_frame <= 51
    plot: [start_frame, end_frame)
    0 = data[:,0,:,:] 
    0 = pred[:,0,:,:]
    '''
    if not isinstance(sample, list):
        sample = [sample]
    for sample_idx in sample:
        pred = batch_of_pred[sample_idx].detach().to(torch.device('cpu')).squeeze()
        target = batch_of_target[sample_idx].detach().to(torch.device('cpu')).squeeze()
        num_frames = end_frame-start_frame
        fig, axs = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))
        for frame in range(start_frame, end_frame):
            axs[0, frame-start_frame].imshow(target[frame,:,:], cmap="Greys")
            axs[0, frame-start_frame].axis('off')
            axs[1, frame-start_frame].imshow(pred[frame,:,:], cmap="Greys")
            axs[1, frame-start_frame].axis('off')
        plt.savefig(f'frames_in_sample_{sample_idx}.png', dpi=120)
        plt.close()

def plot_curve(idx, vector, save_path, filename):
    idx = idx.detach().to(torch.device('cpu')).squeeze()
    vector = vector.detach().to(torch.device('cpu')).squeeze()
    fig, axs = plt.subplots(1,1)
    axs.plot(idx, vector)
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
        '''specify where to save the figure and with what variable name'''
        mat_name = mat_name.replace(' ','_')
        make_dir(f"{folder_log}/"+mat_name)
        self.save_folder = f"{folder_log}/"+mat_name
        self.mat_name = mat_name

    def plot(self, mat, epoch=None):
        '''pass the matrix tensor and plot (doesn't take index yet)'''
        if mat.dim() == 3:
            mat_list = [mat[idx_unit,:,:].squeeze().cpu() for idx_unit in range(mat.shape[0])] 
        else:
            mat_list = [mat.cpu()]
        w_h_ratio = (mat_list[0].shape[1]/mat_list[0].shape[0])
        if w_h_ratio >= 1:
            fig, axs = plt.subplots(1, len(mat_list), figsize=(0.75+2*len(mat_list)*w_h_ratio, 2))
            cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7]) # left bottom width height
        else:
            fig, axs = plt.subplots(1, len(mat_list), figsize=(2*len(mat_list), 0.75+2/(w_h_ratio)))
            cbar_ax = fig.add_axes([0.10, 0.85, 0.8, 0.03]) # left bottom width height
            cbar_ax.orientation = 'horizontal'
        for idx_mat, mat in enumerate(mat_list):
            if len(mat_list) == 1:
                im=axs.imshow(mat, cmap='Greys', interpolation='nearest')
            else:
                im=axs[idx_mat].imshow(mat, cmap='Greys', interpolation='nearest')
        
        fig.colorbar(im, cax=cbar_ax)
        if epoch is not None:
            fig.suptitle(self.mat_name.replace("_", " ")+ f' in epoch [{epoch}]')
            plt.savefig(self.save_folder + '/' + self.mat_name + f'_epoch_{epoch}.png', dpi=120)
        else:
            fig.suptitle(self.mat_name.replace("_", " "))
            plt.savefig(self.save_folder + '/' + self.mat_name + f'.png', dpi=120)
        plt.close()


class ScalarLog:
    def __init__(self, folder_log, var_name, epoch=None):
        self.save_folder = f"{folder_log}/"+var_name
        self.var_name = var_name
        make_dir(self.save_folder)
        self.var = []
        self.idx = []
        self.epoch = epoch

    def reset(self):
        self.var = []
        self.idx = []

    def append(self, value, idx=None):
        self.var.append(value)
        if self.idx == []:
            if idx is None:
                self.idx = [0]
            else:
                self.idx = [idx]
        else:
            if idx is None:
                self.idx.append(self.idx[-1]+1)
            else:
                self.idx.append(idx)

    def save(self):
        var_tensor = torch.tensor(self.var)
        idx_tensor = torch.tensor(self.idx)
        if self.epoch is None:
            torch.save([idx_tensor, var_tensor], self.save_folder +'/'+ self.var_name + '.pt')
        else:
            torch.save([idx_tensor, var_tensor], self.save_folder +'/'+ self.var_name + f'_epoch_{self.epoch}.pt')

class VectorLog:
    def __init__(self, folder_log, var_name, epoch=None):
        '''
        log a vector in a list. vector is supposed to be 1-dim 
        '''
        self.save_folder = f"{folder_log}/"+var_name
        self.var_name = var_name
        make_dir(self.save_folder)
        self.var_stack = None
        self.idx = []
        self.epoch = epoch

    def reset(self):
        self.var_stack = None
        self.idx = []

    def append(self, vector, idx=None):
        if self.var_stack is None:
            self.var_stack = vector.detach().unsqueeze(1)
        else:
            self.var_stack = torch.cat((self.var_stack, vector.detach().unsqueeze(1)), 1)

        if self.idx == []:
            if idx is None:
                self.idx = [0]
            else:
                self.idx = [idx]
        else:
            if idx is None:
                self.idx.append(self.idx[-1]+1)
            else:
                self.idx.append(idx)        

    def save(self):
        idx_tensor = torch.tensor(self.idx)
        if self.epoch is None:
            torch.save([idx_tensor, self.var_stack], self.save_folder +'/'+ self.var_name + '.pt')
        else:
            torch.save([idx_tensor, self.var_stack], self.save_folder +'/'+ self.var_name + f'_epoch_{self.epoch}.pt')

class SaliencyMap():
    def __init__(self, group_rnn: torch.nn.Module, args: argparse.Namespace) -> None:
        self.model = group_rnn
        self.args = args
        self.saliency_maps: Optional[List[Tensor]] = None

    def differentiate(self, x: Tensor, h_prev: Tensor, abs: bool=True) -> Tensor:
        '''
        h_prev  : (batch_size, num_units, hidden_dim)
        attn_score  : (BS, num_units)
        x       : (batch_size, height, width) / (BS, 1, H, W)
        'h_new' : (batch_size, num_units)

        saliency_maps   : (batch_size, num_units, height, width)]
        '''
        x = x.clone().requires_grad_(True)
        h_prev = h_prev.clone()
        output, h_new, intm = self.model(x, h_prev)
        attn_score = intm['input_attn'] # ()
        h_new_out = h_new # save a copy to return
        # h_new = torch.norm(h_new, p=2, dim=2) # (_, _, _,) -> (_, _,)
        saliency_maps: List[Tensor] = []
        mask_init = torch.zeros(1, h_prev.shape[1]).to(x.device)
        for module_idx in range(h_prev.shape[1]):
            mask = mask_init
            mask[:, module_idx] = 1.
            if x.grad is not None:
                x.grad = torch.zeros_like(x.grad)
            (attn_score*mask).backward(gradient=torch.ones_like(attn_score), retain_graph=True)
            saliency_maps.append(x.grad.unsqueeze(1))
        saliency_maps = torch.cat(saliency_maps, dim=1)
        self.inputs = x.squeeze().cpu() # derivative is x-dependent! 
        self.saliency_maps = saliency_maps.cpu().squeeze()
        if abs:
            self.saliency_maps = torch.abs(self.saliency_maps)
        return output, h_new_out, intm

    def plot(self, 
        sample_indices: Union[List[int], int],
        variable_name: Optional[str]='saliency_hid2inp',
        index_name: Optional[str]=None,
        index: Optional[Union[List[Any], Any]]=None,
        save_folder: Optional[str]='.'
        ) -> None:
        '''
        self.inputs: (BS, H, W)
        self.saliency_maps: (BS, num_units, H, W)
        background: (height, width)
        '''
        num_units = self.saliency_maps.shape[1]
        
        if not isinstance(sample_indices, list):
            sample_indices = [sample_indices]
        for sample_idx in sample_indices:
            sa_map_ = self.saliency_maps[sample_idx] 
            bg_ = self.inputs[sample_idx].unsqueeze(0).repeat((num_units, 1, 1))
            if index_name is not None:
                ind_name_ = index_name+f'_{index}_'+f'sample_{sample_idx}'
            else:
                ind_name_ = f'sample_{sample_idx}'
            plot_saliency(
                background=bg_.detach(),
                saliency=sa_map_,
                variable_name=variable_name,
                index_name=ind_name_,
                index=sample_idx,
                save_folder=save_folder,
                bg_alpha=1,
                sa_alpha=0.7
            )
            

def plot_matrix(matrix: Tensor, 
        matrix_name: Optional[str]='matrix', 
        index_name: Optional[str]=None, 
        index: Optional[Any]=None,
        save_folder: Optional[str]='.') -> None:
    '''
    matrix: (N, num_row, num_col) / (num_row, num_col)
    label: len(label) == N / 1 NONO! should be one! for the whole figure!!
    '''
    assert isinstance(matrix, Tensor)
    if matrix.dim() == 3:
        N = matrix.shape[0]
    elif matrix.dim() == 2:
        N = 1
        matrix = matrix.unsqueeze(0)
    else: 
        raise ValueError('matrix should be 2-dim or 3-dim. ')

    matrix = matrix.cpu()
    w_h_ratio = (matrix[0].shape[1]/matrix[0].shape[0])
    if w_h_ratio >= 1:
        fig, axs = plt.subplots(1, len(matrix), figsize=(0.75+2*len(matrix)*w_h_ratio, 2))
    else:
        fig, axs = plt.subplots(1, len(matrix), figsize=(2*len(matrix), 0.75+2/(w_h_ratio)))

    for subplot_idx, mat_ in enumerate(matrix):
        if len(matrix) == 1:
            im=axs.imshow(mat_, cmap='Greys', interpolation='nearest')
            axs.tick_params(labelleft=False, labelbottom=False)
            divider = make_axes_locatable(axs)
        else:
            im=axs[subplot_idx].imshow(mat_, cmap='Greys', interpolation='nearest')
            axs[subplot_idx].tick_params(labelleft=False, labelbottom=False)
            divider = make_axes_locatable(axs[subplot_idx])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation='horizontal')
    
    if index is not None:
        fig_title = matrix_name.replace("_", " ")+ f' in '+index_name+f' [{index}]'
        filename = save_folder + \
            '/' + matrix_name.replace(' ','_') + '/' + \
            matrix_name.replace(' ','_') + '_' + index_name + \
            f'_{index}.png'
    else:
        fig_title = matrix_name.replace("_", " ").capitalize()
        filename = save_folder + \
            '/' + matrix_name.replace(' ','_') + '/' + \
            matrix_name.replace(' ','_') + '.png'
    fig.suptitle(fig_title)
    make_dir(save_folder+'/'+matrix_name.replace(' ','_'))
    plt.savefig(filename, dpi=120)
    plt.close()

def plot_saliency(
    background: Tensor, 
    saliency: Tensor, 
    variable_name: Optional[str]='variable', 
    index_name: Optional[str]=None, 
    index: Optional[Union[List[Any], Any]]=None,
    save_folder: Optional[str]='.',
    bg_alpha: float=1,
    sa_alpha: float=0.3
) -> None:
    ''' Saliency Map Plot Function: plot M WxH images in a row
    background  : (M, W, H) image in the background 
    saliency    : (M, W, H) saliency on top of the background
    (optional) variable_name: name of variable
    (optinoal) index_name   : put an label on the figure title
    (optinoal) index    : corresponding number in label
    (optional) save_folder  : directory to save plots
    '''
    assert isinstance(background, Tensor)
    assert isinstance(saliency, Tensor)
    assert background.shape == saliency.shape
    if background.dim() == 3:
        N = background.shape[0]
    elif background.dim() == 2:
        N = 1
        background = background.unsqueeze(0)
        saliency = saliency.unsqueeze(0)
    else: 
        raise ValueError('tensor should be 2-dim or 3-dim. ')

    background = background.cpu()
    saliency = saliency.cpu()
    w_h_ratio = (background[0].shape[1]/background[0].shape[0])
    if w_h_ratio >= 1:
        fig, axs = plt.subplots(1, len(background), figsize=(0.75+2*len(background)*w_h_ratio, 2)) # always plot in a row! just adjust the individual img size!
    else:
        fig, axs = plt.subplots(1, len(background), figsize=(2*len(background), 0.75+2/(w_h_ratio))) 

    for subplot_idx, (bg_, sa_) in enumerate(zip(background,saliency)):
        if len(background) == 1:
            axs.imshow(bg_, cmap='Greys', interpolation='nearest', alpha=bg_alpha)
            axs.tick_params(labelleft=False, labelbottom=False)
            sa_im = axs.imshow(sa_, cmap='Reds', alpha=sa_alpha) # default interpolation = 'antialiased'
            divider = make_axes_locatable(axs)
        else:
            axs[subplot_idx].imshow(bg_, cmap='Greys', interpolation='nearest', alpha=bg_alpha)
            axs[subplot_idx].tick_params(labelleft=False, labelbottom=False)
            sa_im = axs[subplot_idx].imshow(sa_, cmap='Reds', alpha=sa_alpha) # default interpolation = 'antialiased'
            divider = make_axes_locatable(axs[subplot_idx])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        fig.colorbar(sa_im, cax=cax, orientation='horizontal')

    if index is not None:
        fig_title = variable_name.replace("_", " ")+ f' in '+index_name+f' [{index}]'
        filename = save_folder + \
            '/' + variable_name.replace(' ','_') + '/' + \
            variable_name.replace(' ','_') + '_' + index_name + \
            f'_{index}.png'
    else:
        fig_title = variable_name.replace("_", " ").capitalize()
        filename = save_folder + \
            '/' + variable_name.replace(' ','_') + '/' + \
            variable_name.replace(' ','_') + '.png'
    fig.suptitle(fig_title)
    make_dir(save_folder+'/'+variable_name.replace(' ','_'))
    plt.savefig(filename, dpi=120)
    plt.close()



def array_handler(data: ArrayLike) -> ArrayLike:
    if isinstance(data, Tensor):
        data = data.detach().to('cpu')
        return data
    elif isinstance(data, ArrayLike):
        return data
    else:
        raise RuntimeError('unrecognized data type: {}'.format(type(data)))

def plot_heatmap(data: ArrayLike, 
                 x_label: Optional[Union[str, List[str]]] = None,
                 y_label: Optional[Union[str, List[str]]] = None,
                 vmin: float=0.,
                 vmax: float=1.,
                 cmap: Optional[Colormap]=None,
                 title: Optional[str]=None,
                 figtitle: Optional[str]=None,
                 figsize: Union[FigSize, str]='auto', cbar: bool=True, annot: Union[bool, str]='auto', fmt: str='.2f', square: bool=True,
                 linewidth: float = .5) -> plt.Figure:
    """
    data: 2D array-like or N x 2D array-like
    cbar: bool
    annot: bool or str
    """
    data = array_handler(data)
    if data.dim() == 3:
        data_list = [_data for _data in data]
        N, H, W = data.shape
    else:
        data_list = [data]
        H, W = data.shape
        N = 1
    if isinstance(annot, bool):
        ...
    else:
        assert isinstance(annot, str)
        if annot == 'auto':
            if H < 6 and W < 6:
                annot = True
            else:
                annot = False
        else:
            raise RuntimeError("'annot': {} unrecognized. ".format(annot))
    if isinstance(figsize, str):
        if figsize == 'auto':
            figsize = ((2+2.5*W/H)*N, 4.5) # W, H
        else:
            raise RuntimeError("'figsize': {} unrecognized. ".format(figsize))
    if N == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(ncols=N, figsize=figsize)
    if cmap is None:
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
    for idx, ax in enumerate(axes):
        data = data_list[idx]
        cax = None
        if cbar and idx == len(data_list)-1:
            fig.subplots_adjust(right=0.85)
            cax = fig.add_axes([0.9, 0.1, 0.03/N, 0.7])
        sns.heatmap(data, 
                    annot=annot, 
                    cbar=cbar and idx == len(data_list)-1,
                    cbar_ax=cax,
                    fmt=fmt, 
                    linewidths=linewidth, 
                    ax=ax, 
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    square=square)
        if x_label is not None:
            if isinstance(x_label, str):
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(x_label[idx])
        if y_label is not None:
            if isinstance(y_label, str):
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(y_label[idx])
        if x_label is None and y_label is None:
            ax.set_axis_off()
        if title is not None:
            if isinstance(title, str):
                ax.set_title(title)
            else:
                ax.set_title(title[idx])
    if figtitle is not None:
        fig.suptitle(figtitle)
    plt.close()
    return fig
class VecStack():
    '''stack T D-dim vectors to T-by-N image
    shapes:
        vec: (batchsize=N, dim=D) 
        img: (batchsize=N, dim=D, cols=T)

    methods:
        append: 
        reset:
        show:
    '''
    def __init__(self) -> None:
        self.img = None

    def append(self, vec: Tensor) -> Tensor:
        if self.img is None:
            self.img = vec.reshape(vec.shape[0],-1,1)
        else:
            self.img = torch.cat((self.img, vec.reshape(vec.shape[0],-1,1)), dim=2)
        return self.img

    def reset(self) -> None:
        self.img = None
    
    def show(self) -> Union[Tensor, None]:
        return self.img


def make_grid_video(target, prediction=None, return_dim=4) -> torch.Tensor:
    assert target.dim() == 5 # N T C H W
    frames = []
    for frame_idx in range(target.shape[1]):
        if prediction is not None:
            grided_frame = make_grid(
                torch.cat([target[:, frame_idx, :, :, :], prediction[:, frame_idx, :, :, :]], dim=0),
                nrow=target.shape[0],
                pad_value=1,
            ) # [B C H W] -> [C, *H, **W]
        else:
            grided_frame = make_grid(target[:, frame_idx, :, :, :], nrow=target.shape[0], pad_value=1) # [B, C, H, W] -> [C, *H, **W]
        frames.append(grided_frame)
    if return_dim==4:
        return torch.stack(frames, dim=0) # [T, C, H, W]
    else:
        return torch.stack(frames, dim=0).unsqueeze(0) # [N, T, C, H, W]

def mplfig_to_video(figs: list[plt.figure], filename: str, fps: int=3) -> VideoClip:
    make_frame = lambda t: mplfig_to_npimage(figs[t])
    animation = VideoClip(make_frame,)
    
    animation.write_videofile(filename, fps=fps)
    return animation

def mplfig_to_npvideo(figs: list[plt.figure]) -> np.ndarray:
    """
        `figs`: list of matplotlib figures
    """
    make_frame = lambda t: mplfig_to_npimage(figs[t]).transpose(2,0,1) # (C, H, W)
    frames = [make_frame(t) for t in range(len(figs))]
    video = np.stack(frames, axis=0) # (T, C, H, W)
    
    return video

def heatmap_to_video(data: Tensor, *args, **kwargs) -> np.ndarray:
    """
        `data`: Shape [T, N, H, W] or [T, H, W]
        
    """
    if data.dim() == 4:
        if data.shape[1] == 1:
            data = data.squeeze(1)
    elif data.dim() != 3:
        raise RuntimeError('expected data.dim() == 4 or 3, got {}'.format(data.dim()))

    figs = []
    for frame_idx, frame in enumerate(data):
        figs.append(plot_heatmap(frame, *args, figtitle=f'Frame {frame_idx:d}', **kwargs))
    return mplfig_to_npvideo(figs)

def main():
    # data = torch.rand((64,51,1,64,64))
    # pred = torch.rand((64,50,1,64,64))
    # error = torch.randn((100,1)) + torch.arange(100).unsqueeze(1)
    # _t = torch.load('./../data.pt')
    # _t = _t.unsqueeze(2)
    # plot_frames(_t, _t, 0, 18, 6)

    # '''test saliency'''
    # background = torch.round(torch.rand((3, 64, 64)))
    # saliency = torch.zeros_like(background) + torch.rand_like(background)*0.05
    # saliency[1, 32:, 32:] = 0.3
    # saliency[1, 48:, 48:] = 0.8
    # saliency[1, 60:, 60:] = 5
    # plot_saliency(background, saliency, 'test variable', 'epoch', 10, '.')
    
    '''test heatmap'''
    data = torch.rand((6, 64, 64))
    # data = torch.rand((4, 6))
    titles = [f'Slot {i}' for i in range(data.shape[0])]
    fig = plot_heatmap(
        data,
        title=titles,
        cbar=True,
        linewidth=0.,
        figtitle='suptitle example'
    )
    ...


if __name__ == "__main__":
    main()