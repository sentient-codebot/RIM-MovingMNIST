"""Script to parse all the command-line arguments"""
import argparse
import json


def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def str2loss_fn(_str):
    _str = _str.upper()
    if 'MSE' in _str:
        return 'MSE'
    if 'BCE' in _str:
        return 'BCE'
    if 'MAE' in _str:
        return 'MAE'
    raise argparse.ArgumentTypeError('Unrecognized loss function type.')

def argument_parser():
    """Function to parse all the arguments"""

    """Define config parser and parser"""
    config_parser = argparse.ArgumentParser(
        description='Experiment Script',
        add_help=False) # a must because otherwise the child will have two help options
    config_parser.add_argument('--cfg_json', type=str)
    config_parser.add_argument('--experiment_name', type=str)

    parser = argparse.ArgumentParser(parents=[config_parser])

    # Experiment Settings
    parser.add_argument('--id', type=str, default='default',
                        metavar='id of the experiment', help='id of the experiment')
    parser.add_argument('--version', type=int, default=1)

    # Training Settings
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='ADD')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='ADD')
    parser.add_argument('--lr', type=float, default=0.0001,
                        metavar='LR', help='ADD')
    parser.add_argument('--save_frequency', type=int, default=20,
                        metavar='Frequency at which the model is saved',
                        help='Number of training epochs after which model is to '
                             'be saved. -1 means that the model is not'
                             'persisted')
    parser.add_argument('--test_frequency', type=int, default=10,
                        metavar="Frequency at which we log the intermediate variables of the model",
                        help='Just type in a positive integer')
    parser.add_argument('--path_to_load_model', type=str, default="",
                        metavar='Relative Path to load the model',
                        help='Relative Path to load the model. If this is empty, no model'
                             'is loaded.')
    parser.add_argument('--should_resume', type=str2bool, nargs='?',
                        const=True, default=False,
                        metavar='Flag to decide if the previous experiment should be '
                                'resumd. If this flag is set, the last saved model '
                                '(corresponding to the given id is fetched)',
                        help='Flag to decide if the previous experiment should be '
                                'resumd. If this flag is set, the last saved model '
                                '(corresponding to the given id is fetched)',)
    parser.add_argument('--loss_fn', type=str2loss_fn, default='BCE')

    # Model settings
    parser.add_argument('--core', type=str, default='RIM')
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=600, metavar='hsize',
                        help='hidden_size')
    #   RIM settings
    parser.add_argument('--num_units', type=int, default=6, metavar='num_blocks',
                        help='Number_of_units')
    parser.add_argument('--k', type=int, default=4, metavar='topk',
                        help='Number_of_topk_blocks')
    parser.add_argument('--rnn_cell', type=str, default='GRU',
                        metavar='dynamics of RIMCell', help='one of LSTM or GRU')              
    parser.add_argument('--num_input_heads', type=int, default=1,
                        metavar='E', help='num of heads in input attention')
    parser.add_argument('--input_dropout', type=float,
                        default=0.5, metavar='dropout', help='dropout')
    parser.add_argument('--comm_dropout', type=float, default=0.5)

    parser.add_argument('--input_key_size', type=int)
    parser.add_argument('--input_value_size', type=int)
    parser.add_argument('--comm_key_size', type=int)
    parser.add_argument('--comm_value_size', type=int)
    parser.add_argument('--num_comm_heads', type=int, default=4)

    args, left_argv = config_parser.parse_known_args() # if passed args BESIDES defined in cfg_parser, store in left_argv

    if args.cfg_json is not None:
        with open(args.cfg_json) as f:
            json_dict = json.load(f)
        args.__dict__.update(json_dict)

    parser.parse_args(left_argv, args) # override JSON values with command-line values

    if args.rim_dropout < 0 or args.rim_dropout > 1:
        args.do_rim_dropout = False
    else:
        args.do_rim_dropout = True

    args.id = f"{args.experiment_name}_"+ args.core.upper() + f"_{args.num_units}_{args.hidden_size}"+\
        f"_dropout_{args.rim_dropout}"+\
        f"_ver_{args.version}"

    args.folder_save = f"./saves/{args.id}"
    args.folder_log = f"./logs/{args.id}"

    return args

def main():
    args = argument_parser()
    pass

if __name__ == "__main__":
    main()