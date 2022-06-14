"""image correlation, in order to find matching between two images"""
import enum
import torch
from functools import partial
import os
from matplotlib import pyplot as plt
import torchvision

def normalized_corr(input, target, padding = (0, 0)):
    """
    input: [N, K1, C, H, W]
    target: [N, K2, C, H, W]
    
    return:
        normalized corr: [N, K1, K2]
    """
    corr_op = partial(torch.nn.functional.conv2d, padding=padding, stride=1) # operating on two same sized images would result in a 3x3 matrix
    norm_corr_collect = []
    for idx, sample in enumerate(input):
        # sample, shape [K, C, H, W]
        sample_target = target[idx] # sample_target, shape [K1, C, H, W]
        corr_mat = corr_op(sample, sample_target).sum(dim=(-1, -2)) # [K1=components, K2=objects, ]
        in_scale = corr_op(sample, sample).sum(dim=(-1, -2)) # [K1=components, K2=objects, ]
        in_scale = torch.diagonal(in_scale, dim1=0, dim2=1).unsqueeze(1) # [K1, 1, ]
        tar_scale = corr_op(sample_target, sample_target).sum(dim=(-1, -2)) # [K2, K2, ]
        tar_scale = torch.diagonal(tar_scale, dim1=0, dim2=1).unsqueeze(0) # [1, K2, ]
        corr = corr_mat/torch.sqrt(tar_scale)/torch.sqrt(in_scale) # sum and normalize # Shape [K1, K2]
        norm_corr_collect.append(corr)
    norm_corr = torch.stack(norm_corr_collect, dim=0) # Shape [N, K1, K2]
    
    return norm_corr

def main():
    if os.path.exists('./mmnist_sample.pt'):
        mmnist_sample = torch.load('./mmnist_sample.pt')
        labels, input, output, ind_images = *mmnist_sample, 
        # image = input[0].unsqueeze(0).unsqueeze(1).expand(-1, 3, -1, -1, -1) # [N, K, C, H, W]
        image = ind_images[:, 0, ...].unsqueeze(0) # [N, K, C, H, W]
        image = image[:, (1,0), ...]
        image = image + torch.rand_like(image) * 0.5
        target = ind_images[:, 0, ...].unsqueeze(0) # [N, K, C, H, W]
    else:
        image = torch.rand((1, 3, 256, 256))
        target = torch.rand((1, 3, 256, 256))

    corr = normalized_corr(image, target)
        
    fig, axes = plt.subplots(2, max(image.shape[1], target.shape[1]), figsize=(10, 5))
    for idx, ax in enumerate(axes[0]):
        ax.imshow(
            image[0, idx, ...].permute(1, 2, 0).cpu(),
            cmap='gray'
        )
    for idx, ax in enumerate(axes[1]):
        if idx >= target.shape[1]:
            break
        ax.imshow(
            target[0, idx, ...].permute(1, 2, 0).cpu(),
            cmap='gray'
        )

    print(corr)
    plt.show()
    ...
    plt.close()
    
if __name__ == '__main__':
    main()