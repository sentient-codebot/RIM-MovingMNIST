import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Any

from group_operations import GroupLinearLayer, GroupTorchGRU, GroupLSTMCell, SharedWorkspace, SharedBlockGRU, SharedBlockLSTM
from attentions import InputAttention, CommAttention, SparseInputAttention, PositionAttention, SelectionAttention, MultiHeadAttention
from relational_memory import RelationalMemory


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

class SCOFFCell(nn.Module):
    def __init__(self,
                hidden_size,
                input_size,
                num_inputs,
                num_hidden,
                topkval,
                memorytopk,
                step_att,
                num_modules_read_input,
                inp_heads,
                comm_heads,
                do_gru,
                do_rel,
                n_templates,
                share_inp,
                share_inp_attn,
                share_comm_attn,
                memory_slots=4, # used if do_rel
                num_memory_heads=4, # used if do_rel
                memory_head_size=16, # used if do_rel
                memory_mlp=4, # used if do_rel
                attention_out=340, # used if do_rel
                version=1, # always 1
                straight_through_input=False,
                device=None, # used if do_rel
    ):
        super(SCOFFCell, self).__init__()

        self.hidden_size = hidden_size                                  # size of (total) hidden state
        self.num_inputs = num_inputs                                    # = num of input feature vectors
        self.num_hidden = num_hidden                                    # ?
        self.input_size = input_size                                    # = size of feature vector
        self.single_hidden_size = hidden_size // num_hidden             # = individual hidden size
        self.topkval = topkval                                         
        self.memorytopk = memorytopk
        self.step_att = step_att
        self.do_gru = do_gru
        self.do_rel = do_rel
        self.device = device
        self.num_modules_read_input = num_modules_read_input
        self.direct_input = straight_through_input # if True, no input attention is used and input/ofs are associated one to one. 
        if self.direct_input:
            print("parameters block_size_out are not used because no input attention is used")
        self.inp_heads = inp_heads
        self.comm_heads = comm_heads
        # NOTE modified option below
        self.share_inp = share_inp
        print('topk and memorytopk is', self.topkval, self.memorytopk)
        print('input size', self.input_size)
        print('bs out', self.single_hidden_size)
        print('num_modules_read_input', self.num_modules_read_input)
        print('share same input for all object files', self.share_inp)
        print('share inp and comm attn params', share_inp_attn, share_comm_attn)
        print("communication is happening", self.step_att)
        print('defining comm attention')
        self.mha = MultiHeadAttention(n_head=self.comm_heads, d_model_read=self.single_hidden_size, d_model_write=self.single_hidden_size,
                                      d_model_out=self.single_hidden_size, d_k=32, d_v=32,
                                      num_blocks_read=self.num_hidden, num_blocks_write=self.num_hidden,
                                      dropout=0.1, topk=self.num_hidden,n_templates=1,share_comm=share_comm_attn,share_inp=False, grad_sparse=False)


        self.version = version
        assert self.version == 1
        if self.version:
            #It supports the flexibility of each module having a sperate encoder.
            self.inp_att_out = self.single_hidden_size * 1 # not necessairy tho, is the input size for each gru (!= input_size of scoff)
            if not self.direct_input:
                print('defining inp attention')
                self.inp_att = MultiHeadAttention(n_head=self.inp_heads, d_model_read=self.hidden_size//self.num_hidden,
                                            d_model_write=self.input_size,
                                            d_model_out=self.inp_att_out, d_k=64, d_v=self.inp_att_out, 
                                            num_blocks_read=1, # each time only one hidden vector is input, so 1. 
                                            num_blocks_write=self.num_inputs + 1, # num of input feature vectors + one null input
                                            residual=False,
                                            topk=self.num_inputs + 1, n_templates=1, share_comm=False, share_inp=share_inp_attn, grad_sparse=False, skip_write=True)
            else:
                print('no inp attention defined. slots directly fed to schemata with respective OF')
                self.inp_att = None

        else:
            raise ValueError('following lines should NEVER be run! (version=0) it is a cardinal sin.')
            #this is dummy!
            self.inp_att_out = attention_out
            print('Using version 0 att_out is', self.inp_att_out)
            d_v = self.inp_att_out//self.inp_heads
            self.inp_att = MultiHeadAttention(n_head=self.inp_heads, d_model_read=self.single_hidden_size,
                                          d_model_write=self.block_size_in, d_model_out=self.inp_att_out,
                                          d_k=64, d_v=d_v, num_blocks_read=num_hidden, num_blocks_write=self.num_modules_read_input,residual=False,
                                          dropout=0.1, topk=self.num_inputs+1, n_templates=1, share_comm=False, share_inp=share_inp, grad_sparse=False, skip_write=True)


        if do_gru:
            self.block_lstm = SharedBlockGRU(self.inp_att_out*self.num_hidden, self.hidden_size, k=self.num_hidden, n_templates= n_templates)
        else:
  
            self.block_lstm = SharedBlockLSTM(self.inp_att_out*self.num_hidden, self.hidden_size, k=self.num_hidden, n_templates= n_templates)
           

        if self.do_rel:
            raise ValueError("I don't care about using Relational Memory. ")
            memory_key_size = 32
            gate_style = 'unit'
            print('gate_style is', gate_style, memory_slots, num_memory_heads, memory_head_size, memory_key_size, memory_mlp)
            self.relational_memory = RelationalMemory(
                mem_slots=memory_slots,
                head_size=memory_head_size,
                input_size=self.hidden_size,
                output_size=self.hidden_size,
                num_heads=num_memory_heads,
                num_blocks=1,
                forget_bias=1,
                input_bias=0,
                gate_style="unit",
                attention_mlp_layers=memory_mlp,
                key_size=memory_key_size,
                return_all_outputs=False,
            )

            self.memory_size = memory_head_size * num_memory_heads
            self.mem_att = MultiHeadAttention(
                n_head=4,
                d_model_read=self.single_hidden_size,
                d_model_write=self.memory_size,
                d_model_out=self.single_hidden_size,
                d_k=32,
                d_v=32,
                num_blocks_read=self.num_hidden,
                num_blocks_write=memory_slots,
                topk=self.num_hidden,
                grad_sparse=False,
                n_templates=n_templates,
                share_comm=share_comm,
                share_inp=share_inp,
            )

        self.memory=None

    def blockify_params(self):
        self.block_lstm.blockify_params()
    
    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, inp, hx, cx):
        """
        Inputs:
            if self.direct_input:
                if not self.share_inp:
                    `inp`: [batch_size, d_in] -> num_hidden x [batch_size, d_in//num_hidden], split for each block
                else:
                    `inp`: [batch_size, num_inputs, d_in]
            else:
                `inp`: [batch_size, num_inputs, input_size] should be output of slot attention, aka slots
            `hx`: [batch_size, d_hidden]
            `cx`: [batch_size, d_hidden_c]

        Output:
            `hx_new`: [batch_size, d_hidden]
            `cs_new`: [batch_size, d_hidden_c]
            `mask`: mask for inp_use ~ [bs, num_hidden*block_size_out] (transformed from `inp` using attention with `hx`)
            `block_mask`: # [bs, num_hidden, 1]
            `temp_attention`
        """
        batch_size = inp.shape[0]

        inp_use = inp #layer_input[idx_step]
        def _process_input(_input):
            """
                `input`: [BS, num_inputs*input_size]
            """
            _input = _input.unsqueeze(1)

            return torch.cat(
                [_input, torch.zeros_like(_input[:, 0:1, :])], dim=1 # Shape [batch_size, num_inputs+1, ...]
            )

        if self.version:
            if not self.share_inp:
                input_to_attention = [_process_input(_input) for _input in
                            torch.chunk(inp_use, chunks=self.num_hidden, dim=1)
                            ] # [bs, d_in] -> num_hidden x [bs, d_in//num_hidden] -> num_hidden x [bs, 2, d_in//num_hidden]
            else:
                # `inp_use` Shape: [bs, num_inputs, input_size]
                input_to_attention = torch.cat(
                    [inp_use, torch.zeros_like(inp_use[:, 0:1, :])], dim=1
                ) # Shape [bs, num_inputs+1, input_size]

            split_hx = [chunk.unsqueeze(1) for chunk in
                        torch.chunk(hx, chunks=self.num_hidden, dim=1)] # [bs, d_hidden] -> num_hidden x [bs, 1, d_hidden//num_hidden]

            if not self.direct_input:
                if not self.share_inp:
                    output = [self.inp_att(q=_hx, k=_inp, v=_inp) for
                        _hx, _inp in zip(split_hx, input_to_attention)] # num_hidden x ([bs, 1, attn_out], attn, extra_loss); attn_out == block_size_out
                else:
                    output = [self.inp_att(q=_hx, k=input_to_attention, v=input_to_attention) for
                        _hx in split_hx] # num_hidden x ([bs, 1, attn_out], attn, extra_loss). attn_out == block_size_out
                inp_use_list, iatt_list, _ = zip(*output) # num_hidden x ([bs, 1, attn_out], attn, extra_loss)
                inp_use = torch.cat(inp_use_list, dim=1) # [bs, num_hidden, attn_out]
                iatt = torch.cat(iatt_list, dim=1) # [bs, num_hidden, 2]
            else:
                inp_use = inp # Shape: [bs, num_inputs, input_size] NO null input included, but might contain empty slots
                iatt = torch.ones((inp.shape[0], inp.shape[1], 2), device=inp.device) # dummy iatt score. Shape: [bs, num_hidden, 2] 

            inp_use = inp_use.reshape((inp_use.shape[0], self.inp_att_out * self.num_hidden)) # [bs, att_out * num_hidden]

        else:
            raise ValueError('following lines should NEVER be run! (version=0) it is a cardinal sin.')
            #use attention here.
            inp_use = inp_use.reshape((inp_use.shape[0], self.num_inputs, self.block_size_in))
            inp_use = inp_use.repeat(1,self.num_modules_read_input-1,1)
            inp_use = torch.cat([torch.zeros_like(inp_use[:,0:1,:]), inp_use], dim=1)
            batch_size = inp.shape[0]
            inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_hidden, self.single_hidden_size)), inp_use, inp_use)
            iatt = iatt.reshape((self.inp_heads, sz_b, iatt.shape[1], iatt.shape[2]))
            iatt = iatt.mean(0)

            inp_use = inp_use.reshape((inp_use.shape[0], self.inp_att_out*self.num_hidden))



        new_mask = torch.ones_like(iatt[:,:,0]) # Shape: [bs, num_hidden]

        if (self.num_hidden - self.topkval)>0:
            bottomk_indices = torch.topk(iatt[:,:,0], dim=1,
                                sorted=True, largest=True,
                                k = self.num_hidden - self.topkval)[1]

            new_mask.index_put_((torch.arange(bottomk_indices.size(0)).unsqueeze(1), bottomk_indices),
                    torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype))
        mask = new_mask
        memory_inp_mask = mask
        block_mask = mask.reshape((inp_use.shape[0], self.num_hidden,1)) # [bs, num_hidden, 1]
        mask = mask.reshape((inp_use.shape[0],self.num_hidden,1)).repeat((1,1,self.single_hidden_size)).reshape((inp_use.shape[0], self.num_hidden*self.single_hidden_size)) # [bs, num_hidden*block_size_out] mask for inp_use ~ [bs, num_hidden*block_size_out]
        mask = mask.detach()
        memory_inp_mask = memory_inp_mask.detach() # Shape: [bs, num_hidden]


        if self.do_gru:
            hx_new, temp_attention = self.block_lstm(inp_use, hx) # temp_attention: [bs, num_hidden, n_templates]
            cx_new = hx_new
        else:
            hx_new, cx_new, temp_attention = self.block_lstm(inp_use, hx, cx)
        
        hx_old = hx*1.0
        cx_old = cx*1.0 if not self.do_gru else None

        if self.step_att:
            hx_new = hx_new.reshape((hx_new.shape[0], self.num_hidden, self.single_hidden_size))
            hx_new_grad_mask = blocked_grad.apply(hx_new,
                                                  mask.reshape(
                                                      (mask.shape[0],
                                                       self.num_hidden,
                                                       self.single_hidden_size)))
            hx_new_att,attn_out,extra_loss_att = self.mha(hx_new_grad_mask,hx_new_grad_mask,hx_new_grad_mask)
            hx_new = hx_new + hx_new_att

            hx_new = hx_new.reshape((hx_new.shape[0], self.hidden_size))
            extra_loss = extra_loss_att


        hx = (mask)*hx_new + (1-mask)*hx_old # update OFs
        cx = (mask)*cx_new + (1-mask)*cx_old if not self.do_gru else None # update OFs 

        if self.do_rel:
             #memory_inp_mask = new_mask
             batch_size = inp.shape[0]
             memory_inp = hx.view(
                 batch_size, self.num_hidden, -1
             ) * memory_inp_mask.unsqueeze(2)

             # information gets written to memory modulated by the input.
             _, _, self.memory = self.relational_memory(
                 inputs=memory_inp.view(batch_size, -1).unsqueeze(1),
                 memory=self.memory.cuda(),
             )

             # Information gets read from memory, state dependent information reading from blocks.
             old_memory = self.memory
             out_hx_mem_new, out_mem_2, _ = self.mem_att(
                 hx.reshape((hx.shape[0], self.num_hidden, self.single_hidden_size)),
                 self.memory,
                 self.memory,
             )
             hx = hx + out_hx_mem_new.reshape(
                 hx.shape[0], self.num_hidden * self.single_hidden_size
             )

        return hx, cx, mask, block_mask, temp_attention


    def reset_relational_memory(self, batch_size: int):
        self.memory = self.relational_memory.initial_state(batch_size).to(self.device)

    def step_attention(self, hx_new, cx_new, mask):
        hx_new = hx_new.reshape((hx_new.shape[0], self.num_hidden, self.single_hidden_size))
        # bg = blocked_grad()
        hx_new_grad_mask = blocked_grad.apply(hx_new,
                                              mask.reshape((mask.shape[0],
                                                            self.num_hidden,
                                                            self.single_hidden_size)))
        hx_new_att, attn_out, extra_loss_att = self.mha(hx_new_grad_mask, hx_new_grad_mask, hx_new_grad_mask)
        hx_new = hx_new + hx_new_att
        hx_new = hx_new.reshape((hx_new.shape[0], self.hidden_size))
        extra_loss = extra_loss_att
        return hx_new, cx_new, extra_loss

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