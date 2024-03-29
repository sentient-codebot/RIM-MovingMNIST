import numpy as np
import torch
import os
from torch.utils.data import Dataset

def uint8_to_unit(x: np.ndarray):
    return torch.tensor(x/255., dtype=torch.float32)

class SyntheticMOTDataset(Dataset):
    def __init__(self, mode='train', n_steps=10, dataset_class='vmds', transform=uint8_to_unit, root=None, T=0):
        assert dataset_class in ['vmds', 'vor', 'spmot']
        self.transform = transform
        if self.transform is not None:
            print(f'dataset using transform {self.transform}')
        imgs = np.load(os.path.join(root, dataset_class, '{}_{}.npy'.format(dataset_class, mode)))
        imgs = imgs[:, :n_steps]
        if T and T < n_steps:
            imgs = np.concatenate(np.split(imgs, imgs.shape[1]//T, axis=1))
        self.num_samples = len(imgs)
        self.imgs = [img for img in imgs]

    def __getitem__(self, index):
        x = self.imgs.__getitem__(index)
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.num_samples
    
def main():
    trainset = SyntheticMOTDataset(root='../data')
    print(len(trainset))
    sample = next(iter(trainset))
    print(sample)

if __name__ == "__main__":
    main()