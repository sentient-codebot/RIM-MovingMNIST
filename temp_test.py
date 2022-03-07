from RIM import smooth_sign
import torch

med = torch.rand(3, )*0.1 - 20
med.requires_grad_(True)
out = smooth_sign.apply(med)
out.sum().backward()
print(f"med: {med}")
print(f"out: {out}")
print(f"med.grad: {med.grad}")
