import torch
import json
from time import time
from torch.utils.data import Dataset, DataLoader
import os

class SpritesMOT(Dataset):
    """dataset of Sprites-MOT for Multi-object Tracking
    Args:
        
    Returns:
        torch.utils.data.Dataset
    Sample type:
    """
    def __init__(self, root, train=True, download=False):
        super().__init__()
        self.root = root
        self.dataset_path = os.path.join(self.root, 'input')
        self.is_train = train
        with open(os.path.join(self.root, 'data_config.json')) as f:
            self.meta = json.load(f)
        self.file_batch_num = self.meta['train_batch_num'] if self.is_train else self.meta['test_batch_num']
        self.file_batch_size = self.meta['N']
    
    def __repr__(self):
        info = f'<SpritesMOT Dataset({{\n'
        info += f"\ttotal number of samples: {self.__len__()}\n"
        info += f"\tfile batch size: {self.file_batch_size}\n"
        info += "\tsample shape: [%d, %d, %d, %d]\n" % (self.meta['T'], 3, self.meta['H'], self.meta['W'])
        info += f"\tobject size: [{self.meta['h']}, {self.meta['w']}]\n"
        info += f"\timage depth: {self.meta['D']}\n"
        info += f"}})>"
        return info
        
    def __len__(self):
        if self.is_train:
            return self.meta['train_batch_num']*self.meta['N']
        else:
            return self.meta['test_batch_num']*self.meta['N']
        
    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError
        batch_idx = index // self.file_batch_size
        filename = 'train_' if self.is_train else 'test_'
        filename += str(batch_idx) + '.pt'
        data = torch.load(
            os.path.join(
                self.dataset_path,
                filename
            )
        ) # type: torch.Tensor
        # return torch.tensor(data=[index, batch_idx, index % self.file_batch_size])
        return data[index % self.file_batch_size]/255.
    
def main():
    train_set = SpritesMOT(
        root = './data/sprite/train',
        train = False,
        download = False
    )
    print(train_set)
    # for idx, sample in enumerate(train_set):
    #     # print(idx, sample.shape)
    #     ...
        
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )
    for batch_idx, batch in enumerate(train_loader):
        starttime = time()
        print(batch_idx, batch)
        batch_fetch_time = time() - starttime
        print(f'batch fetch time: {batch_fetch_time}')
    

if __name__ == '__main__':
    main()