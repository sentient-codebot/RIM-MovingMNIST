import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp
from torch.nn.utils.rnn import PackedSequence
from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial

from collections import namedtuple
from abc import ABC, abstractmethod

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

class AlphaFix(torch.autograd.Function):
    """
    given: attention_probs, alpha
    perform: a fix on the probs
    """

    @staticmethod
    def forward(ctx, attention_probs, alpha):
        not_null_probs = attention_probs[:,:,0:-1] * alpha.reshape(1,-1,1)
        null_probs = 1 - alpha.reshape(1,-1) + alpha.reshape(1,-1) * attention_probs[:,:,-1] 

        out_probs = torch.cat((not_null_probs, null_probs.unsqueeze(2)), 2)

        ctx.save_for_backward(not_null_probs, null_probs, alpha)

        return out_probs

    @staticmethod
    def backward(ctx, grad_output): # grad_output means the gradient w.r.t. output
        not_null_probs, null_probs, alpha = ctx.saved_tensors

        grad_alpha = torch.cat((not_null_probs, (-1+null_probs).unsqueeze(2)), 2)

        grad_probs = alpha.reshape(1,-1)

        return grad_output * grad_probs, grad_output * grad_alpha

class GroupLinearLayer(nn.Module):
    """
    for num_blocks blocks, do linear transformations independently

    self.w: (num_blocks, din, dout)

    x: (batch_size, num_blocks, din)
        -> permute: (num_blocks, batch_size, din)
        -> bmm with self.w: (num_blocks, batch_size, din) (bmm) (num_blocks, din, dout)
                            for each block in range(num_blocks):
                                do (batch_size, din) mat_mul (din, dout)
                                concatenate
                            result (num_blocks, batch_size, dout)
        -> permute: (batch_size, num_blocks, dout)

    """
    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """
    def __init__(self, inp_size, hidden_size, num_lstms):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        
        self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
        self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hid_state):
        """
        input: x (batch_size, num_lstms, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, num_lstms, hidden_state)
                c ((batch_size, num_lstms, hidden_state))
        """
        h, c = hid_state
        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :,  :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :,  3 * self.hidden_size:].tanh()
        i_t = gates[:, :,  :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t) 
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())#.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
        input: x (batch_size, num_grus, input_size)
               hidden (batch_size, num_grus, hidden_size)
        output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy


class GroupTorchGRU(nn.Module):
    '''
    Calculate num_units GRU cells in parallel
    '''
    def __init__(self, input_size, hidden_size, num_units):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_units = num_units
        gru_list = [nn.GRU(input_size=self.input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False) for _ in range(num_units)]
        self.grus = nn.ModuleList(gru_list)

    def forward(self, inputs, hidden):
        """
        input: x (batch_size, num_units, input_size)
               hidden (batch_size, num_units, hidden_size)
        output: hidden (batch_size, num_units, hidden_size)
        """

        hidden_list = [gru(inputs[:,i,:].unsqueeze(1), hidden[:,i,:].unsqueeze(0).contiguous())[1].squeeze(0) for i, gru in enumerate(self.grus)]
        # hidden_list: list of (batch_size, hidden_size)
        hidden_new = torch.stack(hidden_list, dim=1)

        return hidden_new

class Attention(nn.Module):
    """
    Input:  key_var     (N, num_keys, d_k) used to construct keys
            value_var   (N, num_keys, D_v)
            query_var   (N, num_queries, D_key=D_query)

            x (batch_size, 2, input_size) [The null input is appended along the first dimension]
            h (batch_size, num_units, hidden_size)
    Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
            mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
    """
    def __init__(self,dropout,):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def dot_product_sum(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.kdim)
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        output, scores, probs = torch.matmul(probs, value) 

    def forward(self, query, key, value):
        output = self.dot_product_sum(query, key, value)

        return output

class InputAttention(Attention):
    def __init__(self, 
        input_size,
        hidden_size, 
        kdim,
        vdim,
        num_heads,
        num_blocks,
        k,
        dropout,
        ):
        super().__init__(dropout)
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.num_blocks = num_blocks
        self.k = k

        self.key = nn.Linear(input_size, num_heads * kdim, bias=False)
        self.value = nn.Linear(input_size, num_heads * vdim, bias=False)
        self.query = GroupLinearLayer(hidden_size, kdim * num_heads, num_blocks)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, h):
        key = self.key(x)
        value = self.value(x)
        query = self.query(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        value = torch.mean(self.transpose_for_scores(value,  self.num_heads, self.vdim), dim = 1)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.kdim) 
        attention_scores = torch.mean(attention_scores, dim = 1)

        mask_ = torch.zeros((x.size(0), self.num_blocks), device=x.device)
        not_null_scores = attention_scores[:,:, 0]
        topk1 = torch.topk(not_null_scores,self.k,  dim = 1)
        batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
        row_to_activate = batch_indices.repeat((1,self.k)) # repeat to the same shape as topk1.indices

        mask_[row_to_activate.view(-1), topk1.indices.view(-1)] = 1
        attention_probs = self.dropout(nn.Softmax(dim = -1)(attention_scores))
        inputs = torch.matmul(attention_probs, value) * mask_.unsqueeze(2)

        with torch.no_grad():
            out_probs = nn.Softmax(dim = -1)(attention_scores)[:,:, 0]

        return inputs, mask_, out_probs

class PriorSampler(nn.Module):
    """
    eta = (num_blocks)
    nu = (num_blocks)

    nu_0, eta_0 -> alpha_0 -> v 
    eta_0+N - nu +1 > 0

    """
    def __init__(self, num_blocks, eta_0, nu_0, N, c):
        self.eta_0 = eta_0
        self.nu_0 = nu_0
        self.num_blocks = num_blocks
        self.c = c
        self.log_beta = nn.Parameter(torch.log(nu_0+1) + 0.01 * torch.randn(num_blocks))
        self.log_alpha = nn.Parameter(torch.log(eta_0-nu_0+1) + 0.01 * torch.randn(num_blocks))

    def forward(self, bs):
        alpha = torch.exp(self.log_beta)
        beta = torch.exp(self.log_alpha)
        self.switch_prior_sampler = Beta(alpha, beta)
        switch_prior = self.switch_prior_sampler.sample(bs).reshape(bs, self.num_blocks)
        # TODO compensate for expectation
        E_alpha = alpha/(alpha+beta).unsqueeze(0).repeat(bs, 1) # (1, num_blocks) * (bs, 1) -> (bs, num_blocks)
        self.v_sampler = Binomial(probs=1-switch_prior.flatten())
        # TODO compensate for expectation
        E_v = E_alpha
        v = self.v_sampler.sample().reshape(bs, self.num_blocks, 1)
        reg_loss = self.reg_loss()

        return v, 1./(E_v + 1e-6), reg_loss
    
    def reg_loss(self):
        nu = torch.exp(self.log_beta) - 1
        eta = torch.exp(self.log_alpha) - 1 + nu
        omega_part_1 = - torch.sum(torch.lgamma(eta-nu+1)-torch.lgamma(nu+1),) #first term, sum over k
        omega_part_2 = torch.sum((eta-nu-self.eta_0+self.nu_0)*(torch.digamma(eta-nu+1)-torch.digamma(eta+2)))
        omega_part_3 = torch.sum((nu-self.nu_0)*(torch.digamma(nu+1)-torch.digamma(eta+2)))
        Omega_c = self.c * (omega_part_1+omega_part_2+omega_part_3)
        return Omega_c
    


class SparseInputAttention(Attention):
    def __init__(self, 
        input_size,
        hidden_size, 
        kdim,
        vdim,
        num_heads,
        num_blocks,
        k,
        dropout,
        eta_0,
        nu_0,
        N
        ):
        super().__init__(dropout)
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.num_blocks = num_blocks
        self.k = k

        self.key = nn.Linear(input_size, num_heads * kdim, bias=False)
        self.value = nn.Linear(input_size, num_heads * vdim, bias=False)
        self.query = GroupLinearLayer(hidden_size, kdim * num_heads, num_blocks)
        self.dropout = nn.Dropout(p = dropout)

        self.prior_sampler = PriorSampler(num_blocks, eta_0, nu_0, 64)

    def forward(self, x, h):
        key = self.key(x)
        value = self.value(x)
        query = self.query(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        value = torch.mean(self.transpose_for_scores(value,  self.num_heads, self.vdim), dim = 1)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.kdim) 
        attention_scores = torch.mean(attention_scores, dim = 1)
        attention_probs = nn.Softmax(dim = -1)(attention_scores)

        not_null_probs = attention_probs[:,:, 0]
        
        mask = torch.ones((h.shape[0], h.shape[1], 1), device=h.device) 
        reg_loss = 0.
        if self.training:
            on_fly_sampler = torch.distributions.binomial.Binomial(p = not_null_probs)
            z = on_fly_sampler.sample(x.shape[0]).reshape(h.shape[0], h.shape[1], 1)
            v, compensate, reg_loss = self.prior_sampler(bs=x.shape[0])
            mask = mask * v * z
        
        attention_probs = self.dropout(attention_probs)
        inputs = torch.matmul(attention_probs * compensate, value) * mask.unsqueeze(2) 

        return inputs,  mask, not_null_probs, reg_loss

class CommAttention(Attention):
    """ h, h -> h 
    """
    def __init__(self, 
        hidden_size,
        kdim,
        num_heads,
        num_blocks,
        dropout
        ):
        super().__init__(dropout)
        self.hidden_size = hidden_size
        self.kdim = kdim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.key = GroupLinearLayer(hidden_size, kdim * num_heads, num_blocks)
        self.query = GroupLinearLayer(hidden_size, kdim * num_heads, num_blocks) 
        self.value = GroupLinearLayer(hidden_size, hidden_size * num_heads, num_blocks)
        self.output_fc = GroupLinearLayer(num_heads * hidden_size, hidden_size, num_blocks)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, h, mask):
        key = self.key(h)
        query = self.query(h)
        value = self.value(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)
        value = self.transpose_for_scores(value, self.num_heads, self.hidden_size)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.kdim)
        probs = nn.Softmax(dim=-1)(scores)

        mask = [mask for _ in range(probs.size(1))]
        mask = torch.stack(mask, dim = 1) # repeat activation mask for each head

        probs = probs * mask.unsqueeze(3) # inactive modules have zero-value query -> no context for them
        probs = self.dropout(probs)

        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context.size()[:-2] + (self.num_heads * self.hidden_size,)
        context = context.view(*new_context_layer_shape) # concatenate all heads
        context = self.output_fc(context) # to be add to current h

        return context


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


# modified part
class OmegaLoss(nn.Module):
    def __init__(self, c, eta_0, nu_0):
        super().__init__()
        self.c = c
        self.eta_0 = eta_0
        self.nu_0 = nu_0

    # nu: BATCH x K, eta_0: scaler, nu_0: scalser
    # maar, nu should be the same for the whole batch (it's parameter)
    def forward(self, eta, nu): 
        omega_part_1 = -torch.sum(torch.lgamma(eta-nu+1)-torch.lgamma(nu+1),) #first term, sum over k
        omega_part_2 = torch.sum((eta-nu-self.eta_0+self.nu_0)*(torch.digamma(eta-nu+1)-torch.digamma(eta+2)))
        omega_part_3 = torch.sum((nu-self.nu_0)*(torch.digamma(nu+1)-torch.digamma(eta+2)))
        Omega_c = self.c * (omega_part_1+omega_part_2+omega_part_3)
        return Omega_c

class SparseRIMCell(RIMCell):
    def __init__(self, 
        device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400, input_query_size = 64,
        num_input_heads = 1, input_dropout = 0.1, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4, comm_dropout = 0.1,
        eta_0 = 1, nu_0 = 1
    ):
        super().__init__(device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size, input_value_size, input_query_size,
            num_input_heads, input_dropout, comm_key_size, comm_value_size, comm_query_size, num_comm_heads, comm_dropout)
        self.eta_0 = eta_0,
        self.nu_0 = nu_0
        
        
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