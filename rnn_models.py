import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Any

from group_operations import GroupLinearLayer, GroupTorchGRU, GroupLSTMCell, SharedWorkspace
from attentions import InputAttention, CommAttention, SparseInputAttention, PositionAttention, SelectionAttention


Ctx = namedtuple('RunningContext',
    [
        'input_attn',
        'input_attn_mask',
        'rule_attn',
        'rule_attn_mask',
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
        device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400,
        num_input_heads = 1, input_dropout = 0.1, use_sw = False, comm_key_size = 32, comm_value_size = 100, num_comm_heads = 4, comm_dropout = 0.1, 
        memory_size = None,
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
        self.input_value_size = input_value_size

        self.comm_key_size = comm_key_size
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

        self.use_sw = use_sw
        self.memory_size = memory_size
        if not self.use_sw:
            self.communication_attention = CommAttention(
                hidden_size, comm_key_size, num_comm_heads, num_units, comm_dropout
            )
        else:
            self.communication_attention = SharedWorkspace(
                write_key_size=comm_key_size,
                read_key_size=comm_key_size,
                memory_size=memory_size,
                hidden_size=hidden_size,
                write_dropout=comm_dropout/2,
                read_dropout=comm_dropout/2,
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

    def forward(self, x, hs, cs = None, M=None, get_intm=False):
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
        if not self.use_sw:
            context = self.communication_attention(h_new, mask.squeeze(2))
            h_new = h_new + context
        else:
            M, h_new = self.communication_attention(M, h_new, mask.squeeze(2))

        # Prepare the context/intermediate value
        ctx = Ctx(input_attn=attn_score,
            input_attn_mask=mask.squeeze(),
            rule_attn=mask.squeeze(), # not applicable here
            rule_attn_mask=mask.squeeze(), # not applicable here
            )

        # Update hs and cs and return them
        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
        return hs, cs, M, ctx

class PackedGRU(nn.Module):
    """pack nn.GRU to conveniently only return variables that I want"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, 1, batch_first=True, bidirectional=False)

    def forward(self, x, hs):
        """
        Inputs:
            `x`: (batch_size, input_size)
            `hs`: (batch_size, hidden_size)
        Output: 
            `hs_new`: (batch_size, hidden_size)
        """
        hs = self.gru(
            x.unsqueeze(1),
            hs.unsqueeze(0).contiguous(),
        )[1].squeeze(0)
        return hs

class FastSCOFF(nn.Module):
    """SCOFF with NPS-like way selection mechanism. 
    All hidden states are updated. 

    Terms:
        a rule = an independent GRU
        a hidden state = a recurrent state vector
    Args:

    
    """
    def __init__(self, 
        device, input_size, hidden_size, num_hidden, num_rules, k, rnn_cell, rule_embedding_size=64, rule_select_key_size=64, input_key_size = 64, input_value_size = 400,
        num_input_heads = 1, input_dropout = 0.1, use_sw = False, comm_key_size = 32, comm_value_size = 100, num_comm_heads = 4, comm_dropout = 0.1
    ):
        super().__init__()
        if comm_value_size != hidden_size:
            #print('INFO: Changing communication value size to match hidden_size')
            comm_value_size = hidden_size
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.num_rules = num_rules
        self.rnn_cell = rnn_cell
        self.key_size = input_key_size
        self.k = k
        self.num_input_heads = num_input_heads
        self.num_comm_heads = num_comm_heads
        self.input_key_size = input_key_size
        self.input_value_size = input_value_size

        self.comm_key_size = comm_key_size
        self.comm_value_size = comm_value_size

        self.rule_embedding_size = rule_embedding_size
        self.rule_embeddings = nn.parameter.Parameter(
            data = torch.randn(self.num_rules, self.rule_embedding_size),
        )
        self.rule_select_key_size = rule_select_key_size
        if self.rnn_cell == 'GRU':
            self.rules = nn.ModuleList(
                [PackedGRU(input_size=self.input_size, 
                            hidden_size=self.hidden_size
                ) for _ in range(self.num_rules)]
            )
            # self.rules = GroupTorchGRU(
            #     input_size=self.input_size,
            #     hidden_size=self.hidden_size,
            #     num_units=self.num_hidden,
            # )
        else:
            raise NotImplementedError("Only GRU rnn_cell is supported")
        
        # attentions
        self.position_attention= PositionAttention(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            kdim=self.input_key_size,
            vdim=self.input_value_size,
            num_heads=self.num_input_heads,
            num_hidden=self.num_hidden,
            dropout=input_dropout,
            epsilon=1e-8
        ) # output shape: [batch_size, num_hidden, input_value_size]

        self.selection_attention = SelectionAttention(
            input_size = self.input_value_size + self.hidden_size, # depends on what is used to construct query for each RIM
            rule_emb_size = self.rule_embedding_size,
            kdim = self.rule_select_key_size,
            normalize=False # gumbel_softmax takes unnormlized probs
        )

        self.use_sw = use_sw
        if not self.use_sw:
            self.communication_attention = CommAttention(
                hidden_size, comm_key_size, num_comm_heads, num_hidden, comm_dropout
            )
        else:
            raise NotImplementedError('SW not implemented')


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

        # Compute input attention, 
        inputs, attn_score = self.position_attention(x, hs) # `inputs` Shape: [N, num_hidden, self.input_value_size]
        # Compute rule selection
        #    use concat[hs, inputs] as query
        #    use rule_embeddings as key
        rule_attn_scores = self.selection_attention(
            torch.cat((hs, inputs), dim = 2), # Shape: [N, num_hidden, self.input_value_size + self.hidden_size]
            self.rule_embeddings
        ) # Shape: [N, num_hidden, num_rules]
        
        # Perform rule selection
        if self.training:
            rule_mask = nn.functional.gumbel_softmax(rule_attn_scores, tau=0.5, hard=True) # sample. Shape: [N, num_hidden, num_rules] (one-hot)
        else:
            rule_mask = RuleArgMax.apply(rule_attn_scores) # no sample, just argmax. Shape: [N, num_hidden, num_rules] (one-hot)
        
        # Compute rules (LSTM or GRU) output
        if cs is not None:
            hs, cs = self.rule_apply(inputs, hs, cs, rule_mask=rule_mask)
        else:
            hs = self.rule_apply(inputs, hs, rule_mask=rule_mask)

        # Compute communication attention
        mask = torch.ones(hs.shape[0], hs.shape[1]).to(hs.device)
        context = self.communication_attention(h_new, mask) # `mask` Shape: [N, num_hidden]
        h_new = h_new + context

        # Prepare the context/intermediate value
        ctx = Ctx(input_attn=attn_score,
            input_attn_mask=mask.squeeze(),
            rule_attn=rule_attn_scores,
            rule_attn_mask=rule_mask.squeeze(),
            )

        # Return updated hs (, cs)
        if cs is not None:
            return hs, cs, None, mask
        return hs, None, None, ctx
    
    def rule_apply(self, inputs, hs, cs=None, rule_mask=None, parallel_input=True):
        """
        for i in range(num_hidden): apply rule

        `self.rules`: nn.ModuleList

        Inputs: 
            `inputs` (batch_size, num_hidden, input_value_size)
            `hs` (batch_size, num_hidden, hidden_size)
            `cs` (batch_size, num_hideen, ...)
            `rule_mask` (batch_size, num_hidden, num_rules)
            `parallel_input`: (bool) setting True will combine batch and num_hidden dimension to speed up computation

        Output: 
            `hs_new`
            `cs_new`
        """
        if rule_mask is None:
            raise RuntimeError("rule_mask is None")
        if not parallel_input:
            inputs = inputs.transpose(0,1) # Shape: [num_hidden, batch_size, input_value_size]
            hs = hs.transpose(0,1) # Shape: [num_hidden, batch_size, hidden_size]
            rule_mask = rule_mask.transpose(0,1) # Shape: [num_hidden, batch_size, num_rules]
            if cs is not None:
                cs = cs.transpose(0,1)
            all_hs_new = []
            all_cs_new = []
            for i in range(self.num_hidden):
                if cs is not None:
                    hs_cs_new = [self.rules[j](inputs[i], (hs[i], cs[i])) for j in range(self.num_rules)]
                    hs_new = torch.stack([hs_cs_new[j][0] for j in range(self.num_rules)], dim=1) # Shape: [batch_size, num_rules, hidden_size]
                    cs_new = torch.stack([hs_cs_new[j][1] for j in range(self.num_rules)], dim=1) # Shape: [batch_size, num_rules, ...]
                    hs_new = torch.sum(hs_new * rule_mask[i].unsqueeze(-1), dim = 1) # Shape: [batch_size, num_rules, hidden_size] -> [batch_size, hidden_size]
                    cs_new = torch.sum(cs_new * rule_mask[i].unsqueeze(-1), dim = 1) # Shape: [batch_size, num_rules, ...] -> [batch_size, ...]
                    all_hs_new.append(hs_new)
                    all_cs_new.append(cs_new)
                else:
                    hs_new = torch.stack([hs_cs_new[j] for j in range(self.num_rules)], dim=1) # Shape: [batch_size, num_rules, hidden_size]
                    hs_new = torch.sum(hs_new * rule_mask[i].unsqueeze(-1), dim = 1) # Shape: [batch_size, num_rules, hidden_size] -> [batch_size, hidden_size]
                    all_hs_new.append(hs_new)
            
            if cs is not None:
                return torch.stack(all_hs_new, dim=1), torch.stack(all_cs_new, dim=1)
            else:
                return torch.stack(all_hs_new, dim=1)
        else:
            batch_size, num_hidden = hs.shape[0], hs.shape[1]
            inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2]) # Shape: [num_hidden*batch_size, input_value_size]
            hs = hs.view(hs.shape[0]*hs.shape[1], hs.shape[2]) # Shape: [num_hidden*batch_size, hidden_size]
            cs = cs.view(cs.shape[0]*cs.shape[1], cs.shape[2]) if cs is not None else None
            rule_mask = rule_mask.view(rule_mask.shape[0]*rule_mask.shape[1], rule_mask.shape[2]) # Shape: [num_hidden*batch_size, num_rules]
            if cs is not None:
                hs_cs_new = [self.rules[j](inputs, (hs, cs)) for j in range(self.num_rules)]
                hs_new = torch.stack([hs_cs_new[j][0] for j in range(self.num_rules)], dim=1) # [BS*K, num_rules, hidden_size]
                cs_new = torch.stack([hs_cs_new[j][1] for j in range(self.num_rules)], dim=1) # [BS*K, num_rules, ...]
                hs_new = torch.sum(hs_new * rule_mask.unsqueeze(-1), dim = 1) # [BS*K, num_rules, hidden_size] -> [BS*K, hidden_size]
                cs_new = torch.sum(cs_new * rule_mask.unsqueeze(-1), dim = 1) # [BS*K, num_rules, ...] -> [BS*K, ...]
                hs_new = hs_new.view(batch_size, num_hidden, hs_new.shape[1]) # [BS, num_hidden, hidden_size]
                cs_new = cs_new.view(batch_size, num_hidden, cs_new.shape[1]) # [BS, num_hidden, ...]
                return hs_new, cs_new
            else:
                hs_new = [self.rules[j](inputs, hs) for j in range(self.num_rules)]
                hs_new = torch.stack(hs_new, dim=1) # Shape: [batch_size*num_hidden, num_rules, hidden_size]
                hs_new = torch.sum(hs_new * rule_mask.unsqueeze(-1), dim = 1) # Shape: [batch_size*num_hidden, num_rules, hidden_size] -> [batch_size*num_hidden, hidden_size]
                hs_new = hs_new.view(batch_size, num_hidden, hs_new.shape[1]) # Shape: [batch_size, num_hidden, hidden_size]
                return hs_new


class RIM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell, num_iterations, **kwargs):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.hidden_size
        self.num_units = num_units
        self.k = k
        self.rnn_cell = rnn_cell
        if rnn_cell == 'LSTM':
            raise NotImplementedError('LSTM not implemented yet')
        
        # Parameters for init (shared by all slots)
        self.rim_mu = torch.nn.parameter.Parameter(
            data = torch.randn(1, 1, self.hidden_size),
        )
        self.rim_log_sigma = torch.nn.parameter.Parameter(
            data = torch.randn(1, 1, self.hidden_size),
        )

        # Network Components
        self.rimcell = RIMCell(
            device = self.device,
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_units = self.num_units,
            k = self.k,
            rnn_cell = self.rnn_cell,
            **kwargs
        )

    def forward(self, x):
        """
        Input : x (batch_size, num_inputs, input_size)
    
        Output: after num_iterations updates: 
                    rim_vectors
        """
        rim_vectors = self.rim_mu + torch.exp(self.rim_log_sigma) * torch.randn(
            x.shape[0], self.num_units, self.hidden_size).to(x.device)
        for i in range(self.num_iterations):
            rim_vectors = self.rimcell(x, rim_vectors) # Shape: [batch_size, num_units, hidden_size]

        return rim_vectors

    

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
            input_attn_mask=mask.squeeze(),
            rule_attn=mask.squeeze(), # not applicable here
            rule_attn_mask=mask.squeeze(), # not applicable here
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

class RuleArgMax(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		idx = torch.argmax(input, 1)
		ctx._input_shape = input.shape
		ctx._input_dtype = input.dtype
		ctx._input_device = input.device
		#ctx.save_for_backward(idx)
		op = torch.zeros(input.size()).to(input.device)
		op.scatter_(1, idx[:, None], 1)
		ctx.save_for_backward(op)
		return op

	@staticmethod
	def backward(ctx, grad_output):
		op, = ctx.saved_tensors
		grad_input = grad_output * op
		return grad_input