from numpy import bmat
from utils.metric import adjusted_rand_index
import torch
import torch.nn.functional as F

# a = torch.randint(0,2,(2,10,))
# a = F.one_hot(a, 3)
a = torch.randn((2,10,3))
# b = torch.randint(0,2,(2,10,3))
b = torch.randn((2,10,3))

mmnist_sample = torch.load('./mmnist_sample.pt')
labels, input, output, ind_images = *mmnist_sample, # ind_images, shape [K, T, C, H, W]

a = ind_images[:, 0, ...].permute(1,2,3,0).flatten(start_dim=0, end_dim=-2).unsqueeze(0)
b = a[..., (1,0)]
b = torch.cat(
    (
        b,
        torch.zeros_like(b[..., -1:])
    ),
    dim=-1
)

value = adjusted_rand_index(a,b)
print(value)