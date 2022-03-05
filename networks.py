import torch
import torch.nn as nn
from RIM import RIMCell, SparseRIMCell, OmegaLoss, LayerNorm, Flatten, UnFlatten, Interpolate
from backbone import GroupDropout
from collections import namedtuple
import numpy as np


Intm = namedtuple('IntermediateVariables',
    [
        'input_attn',
        'input_attn_mask',
        'blocked_dec',
    ])

class MnistModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args['cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        if args['sparse']:
            self.rim_model = SparseRIMCell(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
                args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout'],
                args['a'], args['b'], args['threshold']).to(self.device)
            self.eta_0 = torch.tensor(args['a']+args['b']-2, device=self.device)
            self.nu_0 = torch.tensor(args['b']-1, device=self.device)
            self.regularizer = OmegaLoss(1, self.eta_0, self.nu_0) # 1 for now
        else:
            self.rim_model = RIMCell(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
                args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout']).to(self.device)
            

        self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 10)
        self.Loss = nn.CrossEntropyLoss()


    def to_device(self, x):
        return torch.from_numpy(x).to(self.device) if type(x) is not torch.Tensor else x.to(self.device)

    def forward(self, x, y = None):
        x = x.float()
        
        # initialize hidden states
        hs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
        cs = None
        if self.args['rnn_cell'] == 'LSTM':
            cs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)

        x = x.reshape((x.shape[0],-1))
        xs = torch.split(x, self.args["input_size"], 1)

        # pass through RIMCell for all timesteps
        for x in xs[:-1]:
            hs, cs, nu = self.rim_model(x, hs, cs)
        preds = self.Linear(hs.contiguous().view(x.size(0), -1))

        if y is not None:
            # Compute Loss
            y = y.long()
            probs = nn.Softmax(dim = -1)(preds)
            entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1)) # = -entropy
            loss = self.Loss(preds, y) - entropy # what? should be + entropy
            if self.args['sparse']:
                eta = self.eta_0 + y.shape[0] # eta_0 + N
                loss = loss + self.regularizer(eta, nu)
            return probs, loss
        return preds


    def grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args['cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.hidden_size = args['hidden_size']
        self.lstm = nn.LSTMCell(args['input_size'], self.hidden_size)
        self.Linear = nn.Linear(self.hidden_size, 10)
        self.Loss = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)

    def to_device(self, x):
        return x.to(self.device)

    def forward(self, x, y = None):
        x = x.float()
        hs = torch.randn(x.size(0), self.hidden_size).to(self.device)
        cs = torch.randn(x.size(0), self.hidden_size).to(self.device) 

        x = x.reshape((x.shape[0],-1))
        xs = torch.split(x, self.args["input_size"], 1)
        for x in xs:
            # x_ = torch.squeeze(x, dim = 1)
            hs, cs = self.lstm(x, (hs, cs))
        preds = self.Linear(hs)
        if y is not None:
            y = y.long()
            probs = nn.Softmax(dim = -1)(preds)
            entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1))
            loss = self.Loss(preds, y) - entropy
            return probs, loss
        return preds

    
    def grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm


def sparse_loss(beta, gamma):
    # NOTE: loss is defined for BATCH. so it should be the average across the whole batch
    # beta = batch x K
    # gamma = 1x1
    if beta.dim() > 2:
        raise IndexError('expect beta to be (BatchSize, K)')
    loss_sum = -gamma*torch.sum(beta/(2*gamma*beta-gamma-beta+1)*torch.log(beta/(2*gamma*beta-gamma-beta+1)), dim=1)
    loss = torch.mean(loss_sum)
    return loss

class BallModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.hidden_size * args.num_units # NOTE dimension of encoded input. not clearly mentioned in paper
        self.output_size = args.hidden_size * args.num_units
        self.core = args.core.upper()
        self.sparse = self.args.sparse
        self.get_intm = False

        self.Encoder = self.make_encoder().to(self.args.device)
        self.Decoder = None
        self.make_decoder()

        self.rim_dropout = None
        if self.args.do_rim_dropout:
            self.rim_dropout = GroupDropout(p=args.rim_dropout).to(self.args.device) # TODO later test different probs for different modules

        if self.core == 'RIM':
            if not self.sparse:
                self.rnn_model = RIMCell(
                                        device=self.args.device,
                                        input_size=self.input_size, 
                                        num_units=self.args.num_units,
                                        hidden_size=self.args.hidden_size,
                                        k=self.args.k,
                                        rnn_cell='GRU', # defalt GRU
                                        input_key_size=self.args.input_key_size,
                                        input_value_size=self.args.input_value_size,
                                        input_query_size = self.args.input_query_size,
                                        num_input_heads = self.args.num_input_heads,
                                        input_dropout = self.args.input_dropout,
                                        comm_key_size = self.args.comm_key_size,
                                        comm_value_size = self.args.comm_value_size, 
                                        comm_query_size = self.args.comm_query_size, 
                                        num_comm_heads = self.args.num_comm_heads, 
                                        comm_dropout = self.args.comm_dropout
                ).to(self.args.device)
            else:
                self.rnn_model = SparseRIMCell(
                                        device=self.args.device,
                                        input_size=self.input_size, 
                                        num_units=self.args.num_units,
                                        hidden_size=self.args.hidden_size,
                                        k=self.args.k,
                                        rnn_cell='GRU', # defalt GRU
                                        input_key_size=self.args.input_key_size,
                                        input_value_size=self.args.input_value_size,
                                        input_query_size = self.args.input_query_size,
                                        num_input_heads = self.args.num_input_heads,
                                        input_dropout = self.args.input_dropout,
                                        comm_key_size = self.args.comm_key_size,
                                        comm_value_size = self.args.comm_value_size, 
                                        comm_query_size = self.args.comm_query_size, 
                                        num_comm_heads = self.args.num_comm_heads, 
                                        comm_dropout = self.args.comm_dropout,
                                        eta_0 = 2,
                                        nu_0 = 2,
                                        N = self.args.batch_size
                ).to(self.args.device)

        elif self.core == 'GRU':
            self.rnn_model = nn.GRU(
                                    input_size=self.input_size,
                                    hidden_size=self.args.hidden_size * self.args.num_units,
                                    num_layers=1,
                                    batch_first=True,
            ).to(self.args.device)
        elif self.core == 'LSTM':
            raise ValueError('LSTM Baseline Not Implemented Yet. ')
        else:
            raise ValueError('Illegal RNN Core')

    def make_encoder(self):
        """Method to initialize the encoder"""
        print(self.input_size)
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            Flatten(),
            nn.Linear(2304, self.input_size),
            nn.ELU(),
            LayerNorm(),
        )
    
    def make_decoder(self):
        """Method to initialize the decoder"""
        self.Decoder = nn.Sequential(
            nn.Sigmoid(),
            LayerNorm(),
            nn.Linear(self.output_size, 4096),
            nn.ReLU(),
            LayerNorm(),
            UnFlatten(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        ).to(self.args.device)

    def forward(self, x, h_prev):
        ctx = None
        encoded_input = self.Encoder(x)

        if self.rim_dropout is not None:
            h_prev = self.rim_dropout(h_prev)

        reg_loss = 0.
        if self.core=='RIM':
            if not self.sparse:
                h_new, foo, bar, ctx = self.rnn_model(encoded_input, h_prev)
            else:
                h_new, foo, bar, ctx, reg_loss = self.rnn_model(encoded_input, h_prev)
        elif self.core=='GRU':
            h_shape = h_prev.shape # record the shape
            h_prev = h_prev.reshape((h_shape[0],-1)) # flatten
            _, h_new = self.rnn_model(encoded_input.unsqueeze(1), 
                                        h_prev.unsqueeze(0))
            h_new = h_new.reshape(h_shape)
        elif self.core=='LSTM':
            raise ValueError('LSTM core not implemented yet!')
        
        dec_out_ = self.Decoder(h_new.view(h_new.shape[0],-1))
        blocked_out_ = torch.zeros(1)
        if self.get_intm:
            blocked_out_ = self.partial_blocked_decoder(h_new)

        if ctx is not None:
            intm = Intm(input_attn=ctx.input_attn, 
                input_attn_mask=ctx.input_attn_mask,
                blocked_dec=blocked_out_
                )
        else:
            intm = intm = Intm(input_attn=torch.zeros(1), 
                input_attn_mask=torch.zeros(1),
                blocked_dec=blocked_out_
                )
        
        return dec_out_, h_new, reg_loss, intm

    def init_hidden(self, batch_size): 
        # assert False, "don't call this"
        return torch.rand((batch_size, 
            self.args.num_units, 
            self.args.hidden_size), 
            requires_grad=False)

    def nan_hook(self, out):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            raise RuntimeError(f"Found NAN in {self.__class__.__name__}: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    @torch.no_grad()
    def partial_blocked_decoder(self, h):
        out_list_ = [self.Decoder(h.view(h.shape[0],-1)).unsqueeze(1)] # all-pass
        for block_idx in range(h.shape[1]):
            mask = torch.zeros((h.shape[0],h.shape[1],1), device=self.args.device)
            mask[:, block_idx, :] = 1
            h_masked = h * mask - (1-mask) * 1e-7 # mask==1 -> no change, mask==0 -> 1e-7
            out_ = self.Decoder(h_masked.view(h.shape[0],-1)) # (BS, 1, 64, 64)
            out_list_.append(out_.unsqueeze(1))
        out_ = torch.cat(out_list_, dim=1) # (BS, num_blocks, 1, 64, 64)
        return out_

def clamp(input_tensor):
    return torch.clamp(input_tensor, min=-1e6, max=1e6)

def main():
    gamma = 0.1
    K = 6
    beta = torch.rand(10,6)
    sparse_l = sparse_loss(beta, gamma)
    print(f'sparse regularization loss is {sparse_l}')

if __name__ == "__main__":
    main()

