from logbook.logbook import LogBook
from test_mmnist import setup_model
from utils.util import set_seed
from logbook.logbook import LogBook
from argument_parser import argument_parser

import torch

from datasets.MovingMNIST import MovingMNIST
from utils.visualize import SaliencyMap

set_seed(2022) #Nan is stupid ciao

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
    args.rim_dropout = -1
    args.do_rim_dropout = False
    sa_map = SaliencyMap(model.train(), args)
    data = next(iter(test_loader)).to(args.device)
    if data.dim()==4:
        data = data.unsqueeze(2).float()
    rollout = False
    hidden = model.init_hidden(data.shape[0]).to(args.device).detach()
    for frame in range(data.shape[1]-1):
        if not rollout:
            inputs = data[:, frame, :, :, :].to(args.device)
        elif frame >= 5:
            inputs = output.to(args.device)
        else:
            inputs = data[:, frame, :, :, :]
        output, hidden, intm = sa_map.differentiate(inputs, hidden)
        sa_map.plot(
            sample_indices=0,
            variable_name='saliency_hid2inp',
            index_name='frame',
            index=frame,
            save_folder=args.folder_log+'/intermediate_vars'
        )

if __name__ == "__main__":
    main()