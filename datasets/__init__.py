import torch

from .BouncingBall import BouncingBall
from .MovingMNIST import MovingMNIST
from .SpritesMOT import SpritesMOT
from .SynMOTs import SyntheticMOTDataset

import os
DEBUG = os.environ.get('DEBUG', False)

def mini_dataset(nfold=10):
    """
    Args:
        `nfold`: number of fold on the dataset length
    """
    def mini_dataset_dec(dataset: torch.utils.data.Dataset):
        print(f'Warning: {dataset.__name__} length is divided by {nfold}')
        old_len = dataset.__len__
        dataset.__len__ = lambda args: max(old_len(args) // nfold, 1)
        return dataset
    return mini_dataset_dec

if DEBUG:
    @mini_dataset(nfold=10)
    class BouncingBall(BouncingBall):
        pass

    @mini_dataset(nfold=10)
    class MovingMNIST(MovingMNIST):
        pass
    
    @mini_dataset(nfold=10)
    class SpritesMOT(SpritesMOT):
        pass

def setup_dataloader(args):
    """function to setup dataset and dataloaders
    
    Args:
        `args`: parsed args. 
        
    Retuens:
        (train_loader, val_loader, test_loader) 
        
    """
    val_set = None
    val_loader = None
    if args.task == 'MMNIST':
        train_set = MovingMNIST(
            root=args.dataset_dir, 
            train=True, 
            n_frames_input=10,
            n_frames_output=10,
            num_objects=[1,2],
            download=True
        )
        test_set = MovingMNIST(
            root=args.dataset_dir, 
            train=False, 
            n_frames_input=10,
            n_frames_output=10,
            num_objects=[2],
            download=True
        )
        val_set = MovingMNIST(
            root=args.dataset_dir,
            train=True,
            n_frames_input=10,
            n_frames_output=10,
            num_objects=[1,2,3],
            download=True
        )
    elif args.task == 'BBALL':
        train_set = BouncingBall(root=args.dataset_dir, train=True, length=20, filename=args.ball_trainset)
        test_set = BouncingBall(root=args.dataset_dir, train=False, length=50, filename=args.ball_testset)
    elif args.task == 'TRAFFIC4CAST':
        raise NotImplementedError('Traffic4Cast not implemented')
    elif args.task == 'SPRITESMOT':
        train_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='train',
            dataset_class='spmot',
        )
        val_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='val',
            dataset_class='spmot',
        )
        test_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='test',
            dataset_class='spmot',
        )
    elif args.task == 'VMDS':
        train_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='train',
            dataset_class='vmds',
        )
        val_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='val',
            dataset_class='vmds',
        )
        test_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='test',
            dataset_class='vmds',
        )
    elif args.task == 'VOR':
        train_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='train',
            dataset_class='vor',
        )
        val_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='val',
            dataset_class='vor',
        )
        test_set = SyntheticMOTDataset(
            root=args.dataset_dir, # '.../data'
            mode='test',
            dataset_class='vor',
        )
    else:
        raise ValueError('Unknown task'+args.task)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if not DEBUG else 0,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if not DEBUG else 0,
    )
    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4 if not DEBUG else 0,
        )

    return train_loader, val_loader, test_loader

def main():
    train_set = BouncingBall(root='./data', train=True, length=20, filename='balls4mass64.h5')
    print(len(train_set))
    # train_set = @mini_dataset(nfold=10)(train_set)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    for i, data in enumerate(train_loader):
        print(data.shape)
        # print(target.shape)
        if i == 10:
            break

if __name__ == '__main__':
    main()
