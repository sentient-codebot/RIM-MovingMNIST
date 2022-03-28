import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Any

from group_operations import GroupLinearLayer, GroupTorchGRU, GroupLSTMCell
from attentions import InputAttention, CommAttention, SparseInputAttention


Ctx = namedtuple('RunningContext',
    [
        'input_attn',
        'input_attn_mask'
    ])

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0

class RIMCell(nn.Module):
    def __init__(self, 
        device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400, input_query_size = 64,
        num_input_heads = 1, input_dropout = 0.1, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4, comm_dropout = 0.1
    ):
        super().__init__()
        if comm_value_size != hidden_size:
            #print('INFO: Changing communication value size to match hidden_size')
            comm_value_size = hidden_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_units =num_units
        self.rnn_cell = rnn_cell
        self.key_size = input_key_size
        self.k = k
        self.num_input_heads = num_input_heads
        self.num_comm_heads = num_comm_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        assert input_key_size == input_query_size, "Key and query should be of same size, no? " # they must be equal! 
        self.input_value_size = input_value_size

        self.comm_key_size = comm_key_size
        self.comm_query_size = comm_query_size
        self.comm_value_size = comm_value_size

        if self.rnn_cell == 'GRU':
            # self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
            self.rnn = GroupTorchGRU(input_value_size, hidden_size, num_units) 
        else:
            self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)
        
        # attentions
        self.input_attention_mask = InputAttention(
            input_size, 
            hidden_size, 
            input_key_size, 
            input_value_size,
            num_input_heads, 
            num_units, 
            k,
            input_dropout
        )

        self.communication_attention = CommAttention(
            hidden_size, comm_key_size, num_comm_heads, num_units, comm_dropout
        )


    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def nan_hook(self, out):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    def inf_hook(self, _tensor):
        inf_mask = torch.isinf(_tensor)
        if inf_mask.any():
            raise RuntimeError(f"Found NAN in {self.__class__.__name__}: ", inf_mask.nonzero(), "where:", _tensor[inf_mask.nonzero()[:, 0].unique(sorted=True)])

    def forward(self, x, hs, cs = None, get_intm=False):
        """
        Input : x (batch_size, num_inputs, input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        size = x.size()
        if x.dim() == 2: # Shape: (batch_size, input_size)
            null_input = torch.zeros(size[0], 1, size[1]).float().to(self.device)
            x = torch.cat((x.unsqueeze(1), null_input), dim = 1) # Shape: [batch_size, 1+1, input_size]
        elif x.dim() == 3: # Shape: [batch_size, num_inputs, input_size]
            null_input =  torch.zeros(size[0], 1, size[2]).float().to(self.device)
            x = torch.cat((x, null_input), dim = 1) # Shape: [batch_size, num_inputs+1, input_size]
        else:
            raise RuntimeError("Invalid input size")

        # Compute input attention
        inputs, mask, attn_score = self.input_attention_mask(x, hs)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0
        
        # Compute RNN(LSTM or GRU) output
        
        if cs is not None:
            hs, cs = self.rnn(inputs, (hs, cs))
        else:
            hs = self.rnn(inputs, hs)

        # Block gradient through inactive units
        mask = mask.unsqueeze(2).detach()
        h_new = blocked_grad.apply(hs, mask)

        # Compute communication attention
        context = self.communication_attention(h_new, mask.squeeze(2))
        h_new = h_new + context

        # Prepare the context/intermediate value
        ctx = Ctx(input_attn=attn_score,
            input_attn_mask=mask.squeeze()
            )

        # Update hs and cs and return them
        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
            return hs, cs, None, mask
        return hs, None, None, ctx


class RIM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell, n_layers, bidirectional, **kwargs):
        super().__init__()
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1
        self.rnn_cell = rnn_cell
        self.num_units = num_units
        self.hidden_size = hidden_size
        if self.num_directions == 2:
            self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i < 2 else 
                RIMCell(self.device, 2 * hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers * self.num_directions)])
        else:
            self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i == 0 else
            RIMCell(self.device, hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers)])

    def layer(self, rim_layer, x, h, c = None, direction = 0):
        batch_size = x.size(1)
        xs = list(torch.split(x, 1, dim = 0))
        if direction == 1: xs.reverse()
        hs = h.squeeze(0).view(batch_size, self.num_units, -1)
        cs = None
        if c is not None:
            cs = c.squeeze(0).view(batch_size, self.num_units, -1)
        outputs = []
        for x in xs:
            x = x.squeeze(0)
            hs, cs = rim_layer(x.unsqueeze(1), hs, cs)
            outputs.append(hs.view(1, batch_size, -1))
        if direction == 1: outputs.reverse()
        outputs = torch.cat(outputs, dim = 0)
        if c is not None:
            return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
        else:
            return outputs, hs.view(batch_size, -1)

    def forward(self, x, h = None, c = None):
        """
        Input: x (seq_len, batch_size, feature_size
               h (num_layers * num_directions, batch_size, hidden_size * num_units)
               c (num_layers * num_directions, batch_size, hidden_size * num_units)
        Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
                h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
        """

        hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
        hs = list(hs)
        cs = None
        if self.rnn_cell == 'LSTM':
            cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
            cs = list(cs)
        for n in range(self.n_layers):
            idx = n * self.num_directions
            if cs is not None:
                x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
            else:
                x_fw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None)
            if self.num_directions == 2:
                idx = n * self.num_directions + 1
                if cs is not None:
                    x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
                else:
                    x_bw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None, direction = 1)

                x = torch.cat((x_fw, x_bw), dim = 2)
            else:
                x = x_fw
        hs = torch.stack(hs, dim = 0)
        if cs is not None:
            cs = torch.stack(cs, dim = 0)
            return x, hs, cs
        return x, hs

class SparseRIMCell(RIMCell):
    def __init__(self, 
        device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400, input_query_size = 64,
        num_input_heads = 1, input_dropout = 0.1, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4, comm_dropout = 0.1,
        eta_0 = 1, nu_0 = 1, N = 32
    ):
        super().__init__(device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size, input_value_size, input_query_size,
            num_input_heads, input_dropout, comm_key_size, comm_value_size, comm_query_size, num_comm_heads, comm_dropout)
        self.eta_0 = eta_0,
        self.nu_0 = nu_0
        self.input_attention = SparseInputAttention(
            input_size,
            hidden_size,
            input_key_size,
            input_value_size,
            num_input_heads,
            num_units,
            k,
            input_dropout,
            eta_0,
            nu_0,
            device
        )
        
        
    def forward(self, x, hs, cs = None):
        """
        Input : x (batch_size, input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        size = x.size()
        null_input = torch.zeros(size[0], 1, size[1]).float().to(self.device)
        x = torch.cat((x.unsqueeze(1), null_input), dim = 1)

        # Compute input attention
        inputs, mask, attn_score, reg_loss = self.input_attention(x, hs)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0
        
        # Compute RNN(LSTM or GRU) output
        
        if cs is not None:
            hs, cs = self.rnn(inputs, (hs, cs))
        else:
            hs = self.rnn(inputs, hs)

        # Block gradient through inactive units
        mask = mask.unsqueeze(2).detach() # make a detached copy
        h_new = blocked_grad.apply(hs, mask)
        # mask = mask.unsqueeze(2)
        # h_new = hs

        # 1. Compute communication attention
        context = self.communication_attention(h_new, mask.squeeze(2))
        h_new = h_new + context

        # 2. Update hs and cs and return them
        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
            return hs, cs, None, mask, reg_loss
        
        # Prepare the context/intermediate value
        ctx = Ctx(input_attn=attn_score,
            input_attn_mask=mask.squeeze()
            )

        return hs, None, None, ctx, reg_loss


class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 64, 8, 8)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x