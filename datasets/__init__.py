import torch

from .BouncingBall import BouncingBall
from .MovingMNIST import MovingMNIST

def setup_dataloader(args):
    """function to setup dataset and dataloaders
    
    Args:
        `args`: parsed args. """
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
    elif args.task == 'BBALL':
        train_set = BouncingBall(root=args.dataset_dir, train=True, length=20, filename=args.ball_trainset)
        test_set = BouncingBall(root=args.dataset_dir, train=False, length=50, filename=args.ball_testset)
    elif args.task == 'TRAFFIC4CAST':
        raise NotImplementedError('Traffic4Cast not implemented')
    else:
        raise ValueError('Unknown task'+args.task)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    return train_loader, test_loader