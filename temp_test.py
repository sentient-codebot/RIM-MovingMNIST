import torch
import matplotlib.pyplot as plt
import torchvision

def main():
    _tensor = torch.load('attention_probs_1.pt', map_location='cpu')

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    _tensor = _tensor[0]
    # _tensor = torchvision.utils.make_grid(_tensor).permute(1, 2, 0)
    im = axs.imshow(_tensor)
    fig.colorbar(im, ax=axs)
    plt.savefig('attention_probs_1.png')
    pass

if __name__ == "__main__":
    main()