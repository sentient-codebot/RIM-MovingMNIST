'''
Definitions about logging using Tensorboard
'''
from glob import glob
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: Optional[str]='./runs') -> None:
        super().__init__(
            log_dir=log_dir
        )

    


def main():
    writer = CustomSummaryWriter(log_dir='./runs/demo_exp')
    for i in range(500):
        writer.add_grad_norm(i+torch.rand((1,1,1))*10., i)
        pass

    writer.close()

if __name__ == '__main__':
    main()
