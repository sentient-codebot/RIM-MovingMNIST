import matplotlib.pyplot as plt
import torch

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

def plot_curve(loss):
    loss = loss.detach().to(torch.device('cpu')).squeeze()
    fig, axs = plt.subplots(1,1)
    axs.plot(loss)
    plt.savefig(f"loss_curve.png",dpi=120)

class VectorLog:
    def __init__(self, save_path, var_name):
        self.save_path = save_path
        self.var_name = var_name+".pt"
        self.var = []

    def append(self, value):
        self.var.append(value)

    def save(self):
        var_tensor = torch.tensor(self.var)
        torch.save(var_tensor, self.save_path+"/"+self.var_name)

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