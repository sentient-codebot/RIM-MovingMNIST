import torch
import matplotlib.pyplot as plt
import torchvision

def main():
    pred_dict = torch.load('predictions.pt', map_location='cpu')
    data = pred_dict['data']
    prediction = pred_dict['prediction']

    fig, axs = plt.subplots(2, 20, figsize=(20, 3))
    for i in range(20):
        axs[0, i].xaxis.set_visible(False)
        axs[1, i].xaxis.set_visible(False)
        axs[0, i].yaxis.set_visible(False)
        axs[1, i].yaxis.set_visible(False)
        axs[0, i].imshow(data[0, i, 0, :, :].numpy(), cmap='Greys')
        if i==0:
            axs[1, i].imshow(torch.zeros_like(data[0, i, 0, :, :]).numpy(), cmap='Greys')
        else:
            axs[1, i].imshow(prediction[0, i-1, 0, :, :].numpy(), cmap='Greys')
    plt.savefig('prediction.png')
    pass

if __name__ == "__main__":
    main()