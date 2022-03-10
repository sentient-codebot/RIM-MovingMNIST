import torch
import torch.nn as nn
import math
from group_operations import GroupLinearLayer

from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial



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

        return inputs, mask, not_null_probs, reg_loss

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