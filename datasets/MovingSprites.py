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
        ('train_msprites.pt', '0e862810f6b0a08be91c8e0de6ff48b7'),
        ('test_msprites.pt', '68dd6c454f1ccf6aed2bc63fedcfa361'),
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
            check_integrity(os.path.join(self.root, filename), md5=md5)
            for filename, md5 in self.dataset_files
        )

def main():
    train_set = MovingSprites(
        root='./data',
        train=True,
    )
    print(train_set)
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    for idx, samples in enumerate(tqdm(train_loader)):
        print(samples.shape)
        break
    video = train_set[random.randint(0, len(train_set))] # video.size() = (20, 3, 64, 64)
    show = make_grid(video, nrow=10, pad_value=255)
    plt.imshow(show.numpy().transpose(1, 2, 0))
    plt.show()
    pass


if __name__ == '__main__':
    main()