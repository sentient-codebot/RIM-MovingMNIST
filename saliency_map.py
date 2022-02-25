from logbook.logbook import LogBook
from test_mmnist import setup_model
from utils.util import set_seed
from logbook.logbook import LogBook
from argument_parser import argument_parser

import torch

from data.MovingMNIST import MovingMNIST
from utils.visualize import SaliencyMap

set_seed(1997)

def main():
    args = argument_parser()
    print(args)
    logbook = LogBook(config = args)

    if not args.should_resume:
        args.should_resume = True
    
    cudable = torch.cuda.is_available()
    args.device = torch.device("cuda" if cudable else "cpu")

    model = setup_model(args=args, logbook=logbook)

    args.directory = './data'
    test_set = MovingMNIST(root='./data', train=False, download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True
    )

    sa_map = SaliencyMap(model.eval(), args)
    data = next(iter(test_loader))
    rollout = False
    for frame in range(data.shape[1]-1):
        if not rollout:
            inputs = data[:, frame, :, :, :]
        elif frame >= 5:
            inputs = output
        else:
            inputs = data[:, frame, :, :, :]
        output, hidden, intm = sa_map.differentiate(inputs, hidden)
        sa_map.plot(
            sample_indices=0,
            variable_name='saliency_hid2inp',
            index_name='frame',
            index=frame,
            save_folder=args.folder_log+'intermediate_vars'
        )

if __name__ == "__main__":
    main()