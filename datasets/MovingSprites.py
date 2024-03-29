import gzip
import math
import numpy as np
import os
from tqdm import tqdm
import random
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_and_extract_archive, download_url, check_integrity
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from urllib.error import URLError


def load_fixed_set(root, is_train):
    # Load the fixed dataset
    if is_train:
        filename = 'train_msprites.pt'
    else:
        filename = 'test_msprites.pt'
    path = os.path.join(root, filename)
    dataset = torch.load(path) # dataset: tensor if is_train else (tensor, tensor), shape [N, T, C, H, W], [N, O, T, C, H, W]
    return dataset


class MovingSprites(data.Dataset):
    dataset_files = [
        'train_msprites.pt',
        'test_msprites.pt',
    ]
    def __init__(self, root, train=True, transform=None,):
        '''
        Args:
            `root`: Root directory of the dataset (mnist dataset and moving mnist test set)
            `train`: generate data when is True or `num_objects`!=2, otherwise load the standard test dataset. 
            `transform`: not implemented

        Trainset: 11520 samples of [T=20, C=3, H=64, W=64] video clips
        Testset: 1280 samples of    [T=20, C=3, H=64, W=64] video clips 
                                AND [O=3, T=20, C=3, H=64, W=64] single-object video clips

        Sample shape:
            tuple of:
            `labels`: [num_objects,]
            `input_frames`: [n_frames_input, 1, image_size, image_size]
            `output_frames`: [n_frames_output, 1, image_size, image_size]
        '''
        super().__init__()
        self.root = root
        self.is_train = train
        
        if not self._check_exists():
            raise RuntimeError("Dataset not found. ")

        self.dataset = None
        self.dataset = load_fixed_set(root, train)
        
        self.length = len(self.dataset) if train else len(self.dataset[0])

        self.num_objects = 2
        self.transform = transform

    def __getitem__(self, idx):
        if self.is_train:
            video = self.dataset[idx]
            return video.float() / 255.
        else:
            video = self.dataset[0][idx]
            ind_video = self.dataset[1][idx]
            return video.float() / 255., ind_video.float() / 255.

        if self.transform is not None:
            images = self.transform(images)
            if ind_images is not None:
                ind_images = self.transform(ind_images)

    def __len__(self):
        return self.length

    def _check_exists(self) -> bool:
        return all(
            os.path.exists(os.path.join(self.root, filename))
            for filename in self.dataset_files
        )

def main():
    TRAIN = False
    dataset = MovingSprites(
        root='./data',
        train=True if TRAIN else False,
    )
    print(dataset)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    for idx, samples in enumerate(tqdm(dataloader)):
        # print(samples.shape)
        ...
        break
    samples = list(dataloader)[-1]
    if TRAIN:
        # video = dataset[1249].view(-1, 3, 64, 64) # video.size() = (20, 3, 64, 64)
        video = dataset[torch.randint(0, len(dataset), (4,))].view(-1, 3, 64, 64) # video.size() = (20, 3, 64, 64)
        show = make_grid(video, nrow=20, pad_value=255)
    else:
        # ind_video = samples[1][0].view(-1,3,64,64)
        # ind_video = dataset[1249,][1].view(-1, 3, 64, 64) # video.size() = (20, 3, 64, 64), ind_video.size() = (3, 20, 3, 64, 64)
        foobar = dataset[torch.randint(0, len(dataset), (1,))]
        foo = foobar[0].view(-1, 3, 64, 64) 
        bar = foobar[1].view(-1, 3, 64, 64) # video.size() = (20, 3, 64, 64), ind_video.size() = (3, 20, 3, 64, 64)
        ind_video = torch.cat((foo, bar), dim=0)
        show = make_grid(ind_video, nrow=20, pad_value=255)
    fig, ax = plt.subplots()
    ax.imshow(show.numpy().transpose(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Frames')
    ax.set_ylabel('Samples' if TRAIN else 'Objects')
    plt.savefig('ms_demo.png', dpi=360)
    plt.show()
    
    pass


if __name__ == '__main__':
    main()