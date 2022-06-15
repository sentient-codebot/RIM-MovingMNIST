import torch
import numpy as np
from datasets import MovingMNIST
import argparse
from tqdm import tqdm

# ind_image = np.load('ind_images.npy')

val_set = MovingMNIST(
            root='data',
            train=True,
            n_frames_input=10,
            n_frames_output=10,
            num_objects=[2],# 1 2 3
            download=False,
            length=2000
)

# list_labels, list_input, list_output, list_ind_images = [], [], [], []
# for idx in tqdm(range(len(val_set))):
#     labels,input,output, ind_images = val_set[idx]
#     list_labels.append(labels)
#     list_input.append(input)
#     list_output.append(output)
#     list_ind_images.append(ind_images)

# tensors = torch.stack(list_labels), torch.stack(list_input), torch.stack(list_output), torch.stack(list_ind_images)
# names = ['labels.npy', 'input.npy', 'output.npy', 'ind_images.npy']
# for name, tensor in zip(names, tensors):
#     np.save(name, tensor.numpy())

dataset = []
for idx in tqdm(range(len(val_set))):
    dataset.append(val_set[idx])
    
torch.save(dataset, 'mmnist_val.pt')
...