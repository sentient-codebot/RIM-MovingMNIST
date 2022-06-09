from typing_extensions import runtime
import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Any

from group_operations import GroupLinearLayer, GroupTorchGRU, GroupLSTMCell, SharedWorkspace, SharedBlockGRU, SharedBlockLSTM, SharedGroupGRU
from attentions import InputAttention, CommAttention, SparseInputAttention, PositionAttention, SelectionAttention, MultiHeadAttention

from utils.logging import enable_logging


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
        memory_size = None, use_rule_sharing = False, use_rule_embedding = False, num_rules = None,
        hard_input_attention = False, 
        null_input_type = 'zero',
        input_attention_key_norm = True,
        input_attention_refinement = False,
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
        
        self.use_rule_sharing = use_rule_sharing
        self.use_rule_embedding = use_rule_embedding
        self.num_rules = num_units if num_rules is None else num_rules

        if self.rnn_cell == 'GRU':
            if self.use_rule_sharing:
                self.rnn = SharedGroupGRU(input_value_size, hidden_size, num_units, self.num_rules, use_rule_embedding=self.use_rule_embedding)
            else:
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
            input_dropout,
            hard_argmax=hard_input_attention,
            key_norm=input_attention_key_norm,
            refinement=input_attention_refinement,
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

        self.null_input_type = null_input_type    
        if self.null_input_type == 'zero':   
            self.gen_null_input = torch.zeros
        elif self.null_input_type == 'rand':
            self.gen_null_input = torch.randn
        else:
            print('unrecognized null input type:', self.null_input_type)
            print('using zero null input')
            self.null_input_type = 'zero'
            self.gen_null_input = torch.zeros

        self.do_logging = False
        self.hidden_features = {}

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, hs, cs = None, M=None):
        """
        Input : x (batch_size, num_inputs, input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        size = x.size()
        if x.dim() == 2: # Shape: (batch_size, input_size)
            null_input = self.gen_null_input(size[0], 1, size[1]).float().to(self.device)
            x = torch.cat((x.unsqueeze(1), null_input), dim = 1) # Shape: [batch_size, 1+1, input_size]
        elif x.dim() == 3: # Shape: [batch_size, num_inputs, input_size]
            null_input =  self.gen_null_input(size[0], 1, size[2]).float().to(self.device)
            x = torch.cat((x, null_input), dim = 1) # Shape: [batch_size, num_inputs+1, input_size]
        else:
            raise RuntimeError("Invalid input size")

        # Compute input attention
        inputs, mask, input_attn_probs = self.input_attention_mask(x, hs)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0
        
        # Compute RNN(LSTM or GRU) output 
        rule_attn_gsm = None
        rule_attn_sm = None
        if isinstance(self.rnn, SharedGroupGRU):
            if cs is not None:
                hs, cs, rule_attn_sm, rule_attn_gsm = self.rnn(inputs, (hs, cs))
            else:
                hs, rule_attn_sm, rule_attn_gsm = self.rnn(inputs, hs)
        else:
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
        if self.do_logging:
            self.hidden_features.update(
                {
                    'input_attention_probs': input_attn_probs.detach(), # (0,1), for logging, [N, num_hidden, num_inputs+1]
                    'input_attention_mask': mask.squeeze().detach(), # {0,1}, for logging, [N, num_hidden,]
                }
            )
            if rule_attn_gsm is not None and rule_attn_sm is not None:
                self.hidden_features.update({
                    'rule_attn_probs_sm': rule_attn_sm.detach(), # (0,1), for logging, [N, num_hidden, num_rules]
                    'rule_attn_probs_gsm': rule_attn_gsm.detach(), # {0,1}, for logging, [N, num_hidden, num_rules]
                })

        # Update hs and cs and return them
        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
        return hs, cs, M

class AltSCOFFCell(RIMCell):
    """alternative implementation of SCOFF, by replacing GRU module with SharedGRU in RIMCell"""
    def __init__(self, *args, **kwargs):
        super(AltSCOFFCell, self).__init__(*args, **kwargs)
        if self.rnn_cell == 'LSTM':
            raise NotImplementedError
        elif self.rnn_cell == 'GRU':
            self.rnn = SharedGroupGRU(self.input_value_size, self.hidden_size, self.num_units, self.num_rules, use_rule_embedding=self.use_rule_embedding)
        else:
            raise NotImplementedError
        


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
        ctx = None

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
                hard_input_attention=False,
                null_input_type = 'zero',
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
            raise RuntimeError('input attention is necessary.')
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
            
            # print('defining inp attention')
            # self.inp_att = MultiHeadAttention(n_head=self.inp_heads, d_model_read=self.hidden_size//self.num_hidden,
            #                             d_model_write=self.input_size,
            #                             d_model_out=self.inp_att_out, d_k=64, d_v=self.inp_att_out, 
            #                             num_blocks_read=1, # each time only one hidden vector is input, so 1. 
            #                             num_blocks_write=self.num_inputs + 1, # num of input feature vectors + one null input
            #                             residual=False,
            #                             topk=self.num_inputs + 1, n_templates=1, share_comm=False, share_inp=share_inp_attn, grad_sparse=False, skip_write=True)
            # self.inp_att.attention.query_compeition = True
            print('using custom input attention')
            self.inp_att = InputAttention(
                input_size=self.input_size,
                hidden_size=self.hidden_size//self.num_hidden,
                kdim=64,
                vdim=self.inp_att_out,
                num_heads=self.inp_heads,
                num_hidden=self.num_hidden,
                k=self.topkval,
                dropout=0.1,
                share_query_proj=True,
                num_shared_query_proj=1,
                hard_argmax=hard_input_attention
            )
            print('competition among OFs happening in inp attention')

        else:
            raise ValueError('following lines should NEVER be run! (version=0) it is a cardinal sin.')


        if do_gru:
            self.block_lstm = SharedBlockGRU(self.inp_att_out*self.num_hidden, self.hidden_size, num_hidden=self.num_hidden, n_templates= n_templates)
        else:
  
            self.block_lstm = SharedBlockLSTM(self.inp_att_out*self.num_hidden, self.hidden_size, num_hidden=self.num_hidden, n_templates= n_templates)
           

        if self.do_rel:
            raise ValueError("I don't care about using Relational Memory. ")
        
        self.null_input_type = null_input_type    
        if self.null_input_type == 'zero':   
            self.gen_null_input = torch.zeros_like
        elif self.null_input_type == 'rand':
            self.gen_null_input = torch.randn_like
        else:
            print('unrecognized null input type:', self.null_input_type)
            print('using zero null input')
            self.null_input_type = 'zero'
            self.gen_null_input = torch.zeros

        self.memory=None
        self.do_logging = False
        self.hidden_features = {}

    def blockify_params(self):
        self.block_lstm.blockify_params()
    
    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, inp, hx, cx):
        """
        Inputs:
            if not self.direct_input:
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
                    [inp_use, self.gen_null_input(inp_use[:, 0:1, :])], dim=1
                ) # Shape [bs, num_inputs+1, input_size]

            split_hx = [chunk.unsqueeze(1) for chunk in
                        torch.chunk(hx, chunks=self.num_hidden, dim=1)] # [bs, d_hidden] -> num_hidden x [bs, 1, d_hidden//num_hidden]
            if not isinstance(self.inp_att, InputAttention):
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
                iatt = torch.zeros((inp.shape[0], self.num_hidden, 2)).to(inp.device)
                inp_use, inp_attn_mask_, input_attn_probs = self.inp_att(inp, hx.view(hx.shape[0], self.num_hidden, -1))
                iatt[:,:,0] = 1. - input_attn_probs[:,:,-1]
                if self.do_logging:
                    self.hidden_features['input_attention_probs'] = input_attn_probs.detach()
                    self.hidden_features['input_attention_mask'] = inp_attn_mask_.squeeze().detach()

            inp_use = inp_use.reshape((inp_use.shape[0], self.inp_att_out * self.num_hidden)) # [bs, self.inp_att_out * num_hidden], self.inp_att_out ~= input_size for following GRU

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


        mask = torch.ones_like(iatt[:,:,0]) # Shape: [bs, num_hidden]
        if (self.num_hidden - self.topkval)>0:
            if not isinstance(self.inp_att, InputAttention):
                new_mask = mask        
                bottomk_indices = torch.topk(iatt[:,:,0], dim=1,
                                    sorted=True, largest=True,
                                    k = self.num_hidden - self.topkval)[1]

                new_mask.index_put_((torch.arange(bottomk_indices.size(0)).unsqueeze(1), bottomk_indices),
                        torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype))
            else:
                new_mask = inp_attn_mask_
            mask = new_mask
        block_mask = mask.reshape((inp_use.shape[0], self.num_hidden,1)) # [bs, num_hidden, 1]
        mask = mask.reshape((inp_use.shape[0],self.num_hidden,1)).repeat((1,1,self.single_hidden_size)).reshape((inp_use.shape[0], self.num_hidden*self.single_hidden_size)) # [bs, num_hidden*block_size_out] mask for inp_use ~ [bs, num_hidden*block_size_out]
        mask = mask.detach()


        if self.do_gru:
            hx_new, temp_attention = self.block_lstm(inp_use, hx) # template attention: temp_attention: [bs, num_hidden, n_templates]
            cx_new = hx_new
            if self.do_logging:
                self.hidden_features['rule_attn_probs'] = temp_attention.detach()
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
            raise RuntimeError('no do rel. ')

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