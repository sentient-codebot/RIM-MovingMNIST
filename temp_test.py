import torch
import matplotlib.pyplot as plt

def main():
    attention_probs = torch.load('attention_probs.pt', map_location='cpu')

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    im = axs.imshow(attention_probs[0, :, :])
    fig.colorbar(im, ax=axs)
    plt.savefig('attention_probs.png')
    pass

if __name__ == "__main__":
    main()