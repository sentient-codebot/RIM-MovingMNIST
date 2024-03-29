"""Script to parse all the command-line arguments"""
import argparse
import json
from multiprocessing.sharedctypes import Value
import os
from xmlrpc.client import Boolean


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

def str2encoder(_str):
    _str = _str.upper()
    if _str == 'FLATTEN':
        return 'FLATTEN'
    elif _str == 'NONFLATTEN':
        return 'NONFLATTEN'
    else:
        raise argparse.ArgumentTypeError('Unrecognized encoder type.')

def str2decoder(_str):
    _str = _str.upper()
    if _str == 'CAT_BASIC':
        return 'CAT_BASIC'
    elif _str == 'CAT_SBD':
        return 'CAT_SBD'
    elif _str == 'SEP_BASIC':
        return 'SEP_BASIC'
    elif _str == 'SEP_SBD':
        return 'SEP_SBD'
    else:
        raise argparse.ArgumentTypeError('Unrecognized decoder type.')

def str2task(_str):
    _str = _str.upper()
    if _str == "MOVINGMNIST" or _str == "MMNIST":
        return "MMNIST"
    elif _str in ['MOVINGSPRITES', 'MSPRITES', 'MOVINGSPRITE', 'MSPRITE']:
        return 'MSPRITES'
    elif _str == "BBALL":
        return "BBALL"
    elif _str == "TRAFFIC4CAST":
        return "TRAFFIC4CAST"
    elif _str == "SPRITESMOT":
        return 'SPRITESMOT'
    else:
        raise argparse.ArgumentTypeError('Unrecognized task type.')

def str2ballset(inp_str):
    inp_str = inp_str.lower()
    if inp_str is None:
        return None
    elif inp_str == '4ball':
        return 'balls4mass64.h5'
    elif inp_str == '678ball':
        return 'balls678mass64.h5'
    elif inp_str == 'occlusion' or 'curtain':
        return 'balls3curtain64.h5'
    else:
        raise argparse.ArgumentTypeError('Unrecognized bouncing ball dataset type.')

def str2balltask(inp):
    inp = inp.upper()
    if inp is None:
        print('ball task not specified, using TRANSFER as default.')
        return "TRANSFER"
    elif inp == 'TRANSFER':
        return 'TRANSFER'
    elif True:
        raise NotImplementedError('another type of bouncing ball task not implemented. ')
    else:
        raise argparse.ArgumentTypeError('Unrecognized bouncing ball task type.')

def mmnist_num_obj(string):
    """string: 1,2,3;2;,3"""
    list_ = string.split(';') # ['1,2,3', '2', '3']
    list__ = [foo.split(',') for foo in list_] # [['1', '2', '3'], ['2'], ['3']]
    def _to_int(foo):
        if isinstance(foo[0], list): #[['1', '2', '3']]
            return [_to_int(bar) for bar in foo]
        else: #['1', '2', '3']
            return [int(foo) for foo in foo]
    return _to_int(list__)

def str2inttuple(string):
    out = []
    for s in string:
        try:
            out.append(int(s))
        except ValueError:
            pass
    return tuple(out)

def argument_parser():
    """Function to parse all the arguments"""

    """Define config parser and parser"""
    config_parser = argparse.ArgumentParser(
        description='Experiment Script',
        add_help=False) # a must because otherwise the child will have two help options
    config_parser.add_argument('--cfg_json','--config','--configs', type=str)
    config_parser.add_argument('--experiment_name','--name','--n', type=str)

    parser = argparse.ArgumentParser(parents=[config_parser])

    # Experiment Settings
    parser.add_argument('--id', type=str, default='default',
                        metavar='id of the experiment', help='id of the experiment')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--task', type=str2task, default='MMNIST')
    parser.add_argument('--ball_options', type=str2balltask, default=None, help='options for ball task. transfer or ...')
    parser.add_argument('--ball_trainset', type=str2ballset, default=None, help='train set for ball task')
    parser.add_argument('--ball_testset', type=str2ballset, default=None, help='test set for ball task')
    
    parser.add_argument('--mmnist_num_objects', '--num_objects', '--num_obj', type=mmnist_num_obj, default=[[2],[2],[2]], 
                        help='number of objects in the MMNIST task (train/test/val). default: 2;2;1,2,3')
    parser.add_argument('--enable_tqdm', action='store_true', default=False)

    # Training Settings
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='ADD')
    parser.add_argument('--spotlight_bias',type=str2bool, default=False)
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
    parser.add_argument('--use_val_set', type=str2bool, default=False)
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
    parser.add_argument('--recon_loss_weight', type=float, default=0.1)
    parser.add_argument('--sbd_mem_efficient', type=str2bool, default=False)

    # Model settings
    parser.add_argument('--core', type=str, default='RIM')
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=600, metavar='hsize',
                        help='hidden_size')
    #   encoder settings
    parser.add_argument('--encoder_type', type=str2encoder, default='FLATTEN',
                        help="Type of encoder to use. 'FLATTEN' or 'NONFLATTEN'")
    #   decoder Settings
    parser.add_argument('--decode_hidden', type=str2bool, default=True)
    parser.add_argument('--decoder_type', type=str2decoder, default='CAT_BASIC')
    parser.add_argument('--dec_norm_method', type=str, default='default')
    #   SlotAttention settings
    parser.add_argument('--use_slot_attention', type=str2bool, default=False)
    parser.add_argument('--num_slots', type=int, default=None)
    parser.add_argument("--slot_size", type=int, default=None)
    parser.add_argument("--num_iterations_slot", type=int, default=None)
    parser.add_argument("--load_trained_slot_attention", type=str2bool, default=False)
    parser.add_argument("--use_past_slots", type=str2bool, default=False, help='use past slots as initialization.')
    parser.add_argument('--sa_eval_sm_temp', type=float, default=1.0)
    parser.add_argument('--sa_key_norm', type=str2bool, default=True)
    parser.add_argument('--cell_switch',type=str2inttuple, default=())
    #   RNN common settings
    parser.add_argument('--rnn_cell', type=str, default='GRU',
                        metavar='dynamics of RIMCell/SCOFFCell', help='one of LSTM or GRU')              
    parser.add_argument('--num_hidden', type=int, default=6, metavar='num_blocks',
                        help='Number_of_units')
    parser.add_argument('--k', type=int, default=4, metavar='topk',
                        help='Number_of_topk_blocks') # for RIMs = num active modules, for SCOFF = ?
    parser.add_argument('--num_input_heads', type=int, default=1,
                        metavar='E', help='num of heads in input attention')
    parser.add_argument('--num_comm_heads', type=int, default=4)
    parser.add_argument('--hard_input_attention', type=str2bool, default=False)
    parser.add_argument('--null_input_type', type=str, default='zero', help='either zero or rand')
    parser.add_argument('--input_attention_key_norm', type=str2bool, default=True)
    #       SharedWorkspace settings
    parser.add_argument('--use_sw', type=str2bool, default=False)
    parser.add_argument('--num_sw_write_heads', type=int, default=1)
    parser.add_argument('--num_sw_read_heads', type=int, default=1)
    parser.add_argument('--sw_write_value_size', type=int, default=None)
    parser.add_argument('--sw_read_value_size', type=int, default=None)
    parser.add_argument('--memory_size', type=int, default=None)
    parser.add_argument('--num_memory_slots', type=int, default=None)
    parser.add_argument('--use_memory_for_decoder', type=str2bool, default=False)
    #   SCOFF settings
    parser.add_argument("--num_rules", type=int, default=None)
    parser.add_argument("--slot_straight_input", type=str2bool, default=False, 
                        help="straight through input from slot to ofs without input attention")
    #   RIM settings
    parser.add_argument('--use_rule_sharing', type=str2bool, default=False)
    parser.add_argument('--use_rule_embedding', type=str2bool, default=False)
    parser.add_argument('--do_comm', type=str2bool, default=True)
    # parser.add_argument('--num_rules', type=int, default=None) # already exists for SCOFF
    parser.add_argument('--input_dropout', type=float,
                        default=0.1, metavar='dropout', help='dropout')
    parser.add_argument('--comm_dropout', type=float, default=0.5)
    parser.add_argument('--input_key_size', type=int)
    parser.add_argument('--input_value_size', type=int)
    parser.add_argument('--comm_key_size', type=int)
    parser.add_argument('--comm_value_size', type=int)
    parser.add_argument('--use_compositional_MLP', type=bool, default=False)

    args, left_argv = config_parser.parse_known_args() # if passed args BESIDES defined in cfg_parser, store in left_argv

    if args.cfg_json is not None:
        with open(args.cfg_json) as f:
            json_dict = json.load(f)
        # args.__dict__.update(json_dict) # does not guarantee arg format is correct
        json_argv = []
        for key, value in json_dict.items():
            json_argv.append('--' + key)
            json_argv.append(str(value))
        parser.parse_known_args(json_argv, args)

    parser.parse_args(left_argv, args) # override JSON values with command-line values

    # processing some arguments
    if args.dataset_dir == '':
        if args.task == 'MMNIST':
            args.dataset_dir = '/home/nnan/movingmnist/'
        elif args.task == 'MSPRITES':
            args.dataset_dir = '/home/nnan/msprites/'
        elif args.task == 'BBALL':
            args.dataset_dir = '/home/nnan/BouncingBall/'
        elif args.task == 'TRAFFIC4CAST':
            args.dataset_dir = '/home/nnan/traffic4cast/'
        elif args.task == 'SPRITESMOT':
            args.dataset_dir = '/home/nnan/'
            
    if args.task in ['SPRITESMOT', 'VMDS', 'VOR']:
        if 'SEP' not in args.decoder_type:
            print('Warning: Component-wise decoder has to be used in MOT tasks. Using SEP_BASIC instead.')
            args.decoder_type = str2decoder('SEP_BASIC')
        if args.decode_hidden:
            print('Warning: Reconstruction has to be made for MOT tasks. Setting decode_hidden to False.')
            args.decode_hidden = False
            
    if args.decode_hidden:
        args.recon_loss_weight = 0.0
        print('Decoding hidden state directly, so the reconstruction loss is set to 0.0')
    
    if args.k > args.num_hidden:
        args.k = args.num_hidden
    if args.use_sw:
        if args.memory_size is None:
            args.memory_size = args.num_hidden
        if args.num_memory_slots is None:
            raise ValueError('num_memory_slots must be specified')
        if args.num_sw_write_heads > 1 and args.sw_write_value_size is None:
            raise ValueError('sw_write_value_size must be specified')
        if args.num_sw_read_heads > 1 and args.sw_read_value_size is None:
            raise ValueError('sw_read_value_size must be specified')

    if args.use_slot_attention:
        assert args.num_iterations_slot is not None
        if args.num_slots is None:
            args.num_slots = args.num_hidden
        if args.slot_size is None:
            args.slot_size = args.hidden_size

    if args.task == "BBALL":
        # assert args.ball_options is not None
        if args.ball_options is None:
            args.ball_options = 'TRANSFER'
        assert args.ball_trainset is not None
        assert args.ball_testset is not None

    args.id = f"{args.experiment_name}_"
    if args.use_slot_attention:
        if args.spotlight_bias:
            args.id = args.id + 'SSA_' + f"{args.num_slots}_{args.slot_size}_{args.num_iterations_slot}_"
        else:
            args.id = args.id + 'SA_' + f"{args.num_slots}_{args.slot_size}_{args.num_iterations_slot}_"
    args.id = args.id + args.core.upper() + f"_{args.num_hidden}_{args.hidden_size}"
    if args.core == 'SCOFF':
        args.id = args.id + f"_{args.num_rules}"
    args.id = args.id + f"_ver_{args.version}"
    args.folder_save = f"./saves/{args.id}"
    args.folder_log = f"./logs/{args.id}"
    
    args.mot_pred_file = args.folder_log+'/mot_json.json'
    if args.task == 'SPRITESMOT':
        args.mot_gt_file = os.path.join(args.dataset_dir, 'gt_jsons', 'spmot_test.json')
    elif args.task == 'VMDS':
        args.mot_gt_file = os.path.join(args.dataset_dir, 'gt_jsons', 'vmds_test.json')
    elif args.task == 'VOR':
        args.mot_gt_file = os.path.join(args.dataset_dir, 'gt_jsons', 'vor_test.json')
        
    if args.use_val_set == True and args.task != 'MMNIST':
        raise NotImplementedError

    return args

def main():
    args = argument_parser()
    pass

if __name__ == "__main__":
    main()