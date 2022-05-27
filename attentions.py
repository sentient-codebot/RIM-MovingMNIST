import torch
import torch.nn as nn
import math
from group_operations import GroupLinearLayer

from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial
from torch.distributions.uniform import Uniform

from group_operations import SharedGroupLinearLayer
import numpy as np

from typing import Any


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

    def __init__(self, dropout,):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + \
            (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def dot_product_sum(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-1, -2)
                              ) / math.sqrt(self.kdim)
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        output, scores, probs = torch.matmul(probs, value)

    def forward(self, query, key, value):
        output = self.dot_product_sum(query, key, value)

        return output


class InputAttention(Attention):
    """
    Args:
        `num_blocks`: always equal to number of OFs/hidden state vectors
        `share_query_proj`: whether to share the same projection matrix for all query vectors. 
        `num_shared_query_proj`: number of shared query projection matrices."""

    def __init__(self,
                 input_size,
                 hidden_size,
                 kdim,
                 vdim,
                 num_heads,
                 num_hidden,
                 k,
                 dropout,
                 epsilon=1e-8,
                 share_query_proj=False,
                 num_shared_query_proj=1,
                 hard_argmax=False,
                 key_norm=True,
                 ):
        super().__init__(dropout)
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.num_hidden = num_hidden
        self.k = k

        self.key = nn.Linear(input_size, num_heads * kdim, bias=False)
        self.value = nn.Linear(input_size, num_heads * vdim, bias=False)
        if not share_query_proj:
            # giving each query vector different projection matrix (one-to-one)
            self.query = GroupLinearLayer(
                hidden_size, kdim * num_heads, num_hidden)
        else:
            # giving each query vector different projection matrix (one-to-one)
            self.query = SharedGroupLinearLayer(
                hidden_size, kdim * num_heads, num_hidden)
            # all query share the same projection *proj*
        self.dropout = nn.Dropout(p=dropout)
        self.epsilon = epsilon
        
        self.hard_argmax = hard_argmax
        self.key_norm = key_norm

    def forward(self, x, h):
        key = self.key(x)  # Shape: [batch_size, num_heads, kdim]
        value = self.value(x)
        query = self.query(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        value = torch.mean(self.transpose_for_scores(
            value,  self.num_heads, self.vdim), dim=1)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)

        attention_scores = torch.matmul(
            query, key.transpose(-1, -2)) / math.sqrt(self.kdim)
        attention_scores = torch.mean(attention_scores, dim=1)
        # (batch_size, num_query, num_key) NOTE for each input, rims compete with each other
        attention_probs = nn.Softmax(dim=1)(attention_scores)

        # For each rim, give them normalized summation weights (for each rim, the weights all sum to 1) NOTE is this necessary?
        if self.key_norm:
            attention_probs = attention_probs + self.epsilon  # in case of unstability
            attention_probs = attention_probs / \
                torch.sum(attention_probs, dim=2, keepdim=True)
        if self.hard_argmax:
            attention_probs_mask = (ArgMax.apply(attention_probs)).detach()
            attention_selected_probs = (attention_probs*attention_probs_mask).detach()
            attention_probs = attention_probs*attention_probs_mask/(attention_selected_probs + 0.000001)

        mask_ = torch.zeros((x.size(0), self.num_hidden), device=x.device)
        # Shape: [batch_size, num_blocks, ] NOTE how much focus is NOT on the null input
        not_null_probs = 1. - attention_probs[:, :, -1]
        topk1 = torch.topk(not_null_probs, self.k, dim=1)
        batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
        # repeat to the same shape as topk1.indices
        row_to_activate = batch_indices.repeat((1, self.k))
        mask_[row_to_activate.view(-1), topk1.indices.view(-1)] = 1

        # inputs = (bs, num_blocks, vdim), all value vectors are just scaled version of each other.
        inputs = torch.matmul(self.dropout(
            attention_probs), value) * mask_.unsqueeze(2)

        # with torch.no_grad():
        #     out_probs = 1.-attention_probs[:,:, -1]

        return inputs, mask_, attention_probs


class PositionAttention(Attention):
    def __init__(self,
                 input_size,
                 hidden_size,
                 kdim,
                 vdim,
                 num_heads,
                 num_hidden,
                 dropout,
                 epsilon=1e-8
                 ):
        super().__init__(dropout)
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.num_hidden = num_hidden

        self.key = nn.Linear(input_size, num_heads * kdim, bias=False)
        self.value = nn.Linear(input_size, num_heads * vdim, bias=False)
        self.query = GroupLinearLayer(
            hidden_size, kdim * num_heads, num_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.epsilon = epsilon

    def forward(self, x, h):
        """
        Input:
            `x`: input tensor of shape (batch_size, num_inputs, input_size) -> keys/values
            `h`: hidden state of shape (batch_size, num_hidden, hidden_size) -> queries
        """
        key = self.key(x)  # Shape: [batch_size, num_heads, kdim]
        value = self.value(x)
        query = self.query(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        value = torch.mean(self.transpose_for_scores(
            value,  self.num_heads, self.vdim), dim=1)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)

        attention_scores = torch.matmul(
            query, key.transpose(-1, -2)) / math.sqrt(self.kdim)
        # Shape: (batch_size, num_queries, num_keys)
        attention_scores = torch.mean(attention_scores, dim=1)
        # (batch_size, num_queries, num_keys) NOTE for each query, positions compete with each other
        attention_probs = nn.Softmax(dim=2)(attention_scores)

        output = torch.matmul(self.dropout(attention_probs), value)

        return output, attention_probs


class SelectionAttention(nn.Module):
    """SelectionAttention for selecting rules by matching rules and inputs.

    Args:
        `input_size`    : input size, used to construct queries
        `rule_emb_size` : rule embedding size, used to construct keys
        `kdim`          : dimension of keys
        `normalize`     : [Optional, boolean] whether to normalize the attention scores, default=`True`
        """

    def __init__(self, input_size, rule_emb_size, kdim, normalize=True):
        super().__init__()
        self.input_size = input_size
        self.rule_emb_size = rule_emb_size
        self.kdim = kdim
        self.normalize = normalize
        self.query_proj = nn.Linear(input_size, kdim, bias=False)
        self.key_proj = nn.Linear(rule_emb_size, kdim, bias=False)

    def forward(self, inputs, rule_embeddings):
        """
        Input:
            `inputs`: input tensor of shape (batch_size, num_inputs, input_size)
            `rule_embeddings`: rule embeddings of shape (batch_size, num_rules, rule_emb_size)
        Output:
            `attention_scores`: attention scores of shape (batch_size, num_inputs, num_rules), normalized if `normalize==True`
        """
        query = self.query_proj(inputs)  # Shape: [N, num_inputs, kdim]
        key = self.key_proj(rule_embeddings)  # Shape: [N, num_rules, kdim]
        # Shape: [N, num_inputs, num_rules]
        attention_scores = torch.matmul(
            query, key.transpose(-1, -2)) / math.sqrt(self.kdim)
        if self.normalize:
            # Shape: [N, num_inputs, num_rules]
            attention_scores = nn.Softmax(dim=2)(attention_scores)
        return attention_scores


class PriorSampler():
    """
    eta = (num_blocks)
    nu = (num_blocks)

    nu_0, eta_0 -> alpha_0 -> v 
    eta_0+N - nu +1 > 0

    """

    def __init__(self, num_blocks, alpha_0, beta_0, device):
        super().__init__()
        self.beta_0 = beta_0
        self.alpha_0 = alpha_0
        self.num_blocks = num_blocks
        self.device = device

    def sample(self, log_alpha, log_beta, bs):
        """
        switch' ~ Beta(alpha, beta) (reparameterization) -> switch' = g(phi, alpha, beta), phi ~ new_pdf(.)
        """
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        switch_prior_sampler = Beta(alpha, beta)
        switch_prior = switch_prior_sampler.rsample(
            (bs,)).reshape(bs, self.num_blocks).to(self.device)
        # TODO compensate for expectation
        # (1, num_blocks) * (bs, 1) -> (bs, num_blocks)
        E_alpha = alpha/(alpha+beta).unsqueeze(0).repeat(bs, 1)
        E_v = E_alpha
        u_sampler = Uniform(-1, 0)
        u = u_sampler.sample(switch_prior.shape).to(self.device)
        v = 0.5*smooth_sign.apply(u+switch_prior) + 0.5
        reg_loss = self.reg_loss(log_alpha, log_beta)

        return v, 1./(E_v + 1e-6), reg_loss

    def reg_loss(self, log_alpha, log_beta):
        """
        now implemented as KL divergence
        """
        # nu = torch.exp(self.log_beta) - 1
        # eta = torch.exp(self.log_alpha) - 1 + nu
        # omega_part_1 = - torch.sum(torch.lgamma(eta-nu+1)-torch.lgamma(nu+1),) #first term, sum over k
        # omega_part_2 = torch.sum((eta-nu-self.eta_0+self.nu_0)*(torch.digamma(eta-nu+1)-torch.digamma(eta+2)))
        # omega_part_3 = torch.sum((nu-self.nu_0)*(torch.digamma(nu+1)-torch.digamma(eta+2)))
        # Omega_c = (omega_part_1+omega_part_2+omega_part_3)
        def lbeta_func(alpha, beta): return torch.lgamma(
            alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta)
        beta = torch.exp(log_beta)
        alpha = torch.exp(log_alpha)

        kl = lbeta_func(alpha, beta)-lbeta_func(self.alpha_0, self.beta_0) +\
            (self.alpha_0-alpha)*torch.digamma(self.alpha_0) +\
            (self.beta_0-beta)*torch.digamma(self.beta_0) +\
            (alpha+beta-self.alpha_0-self.beta_0) * \
            torch.digamma(self.alpha_0+self.beta_0)

        return kl.sum()


class icdf_beta(torch.autograd.Function):
    # NOTE automatically implemented by pytorch: Distribution.rsample method
    @staticmethod
    def forward(ctx, x, mask):
        raise NotImplementedError('sorry not yet')
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        cdf_beta(x) = P | icdf_beta(P) = x
        d(icdf_beta(P))/dP = 1 / pdf_beta(x)
        """
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class bernoulli_rsample(nn.Module):
    """ x ~ B(p)
    x = 0.5 * sign(u' + p) + 0.5, u' ~ U(-1,0)
    ctx_x = 0.5 * approx_sign(u + p) + 1, approx_sign(.) = tanh(k*.)
    """

    def __init__(self, p):
        super().__init__()

    def forward(ctx, p, shape):
        assert isinstance(shape, tuple)
        U = torch.distributions.Uniform(-1, 1)
        u = U.sample((*shape, p.shape[0]))
        x = torch.sign(u + p) + 1
        x.requires_grad_(True)
        med = (u+p).detach().requires_grad_(True)
        approx_x = torch.tanh(1.*med)
        ctx.save_for_backward(approx_x, med)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        approx_x, med = ctx.saved_tensors
        approx_x.backward(grad=grad_output)
        return med.grad


class smooth_sign(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)

    def backward(ctx: Any, grad_output: Any) -> Any:
        x, = ctx.saved_tensors
        def scaled_tanh(x): return torch.tanh(100*x)
        func_out, vjp = torch.autograd.functional.vjp(
            scaled_tanh, x, grad_output)
        return vjp


class ArgMax(torch.autograd.Function):
    """forward the hard argmax function, while backward as the soft(arg)max

    Inputs:
        `x`: a tensor of shape `[batch_size, num_slots, num_inputs]`

    Outputs:
        `y`: a one-hot tensor of shape `[batch_size, num_slots, num_inputs]` 
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        indices = torch.argmax(x, dim=-1)  # Shape: [batch_size, num_slots]
        # Shape: [batch_size, num_slots, num_inputs]
        y = torch.nn.functional.one_hot(indices, x.shape[-1])
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        x, = ctx.saved_tensors
        func_out, vjp = torch.autograd.functional.vjp(
            torch.nn.Softmax(dim=-1), x, grad_output)
        return vjp


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
                 device
                 ):
        super().__init__(dropout)
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.num_blocks = num_blocks
        self.k = k

        self.key = nn.Linear(input_size, num_heads * kdim, bias=False)
        self.value = nn.Linear(input_size, num_heads * vdim, bias=False)
        self.query = GroupLinearLayer(
            hidden_size, kdim * num_heads, num_blocks)
        self.dropout = nn.Dropout(p=dropout)

        self.device = device

        self.eta_0 = torch.tensor(eta_0, device=device)
        self.nu_0 = torch.tensor(nu_0, device=device)
        self.beta_0 = self.nu_0+1
        self.alpha_0 = self.eta_0-self.nu_0+1
        self.num_blocks = num_blocks
        self.log_beta = nn.Parameter(
            torch.log(self.nu_0+1) + 0.1 * torch.randn(num_blocks, device=device))
        self.log_alpha = nn.Parameter(torch.log(
            self.eta_0-self.nu_0+1) + 0.1 * torch.randn(num_blocks, device=device))
        self.prior_sampler = PriorSampler(
            num_blocks, self.alpha_0, self.beta_0, device=device)

    def forward(self, x, h):
        key = self.key(x)
        value = self.value(x)
        query = self.query(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        value = torch.mean(self.transpose_for_scores(
            value,  self.num_heads, self.vdim), dim=1)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)

        attention_scores = torch.matmul(
            query, key.transpose(-1, -2)) / math.sqrt(self.kdim)
        attention_scores = torch.mean(attention_scores, dim=1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        not_null_probs = attention_probs[:, :, 0]

        mask = torch.ones((h.shape[0], h.shape[1], 1), device=h.device)
        reg_loss = torch.zeros(1, device=x.device)
        if self.training:
            # implementation 0: differentiable sampler
            # on_fly_sampler = Uniform(-1, 0) # probs (BS, num_blocks)
            # u = on_fly_sampler.sample(h.shape[0:2]).reshape(h.shape[0], h.shape[1], 1).to(x.device)
            # z = 0.5*smooth_sign.apply(u+not_null_probs.unsqueeze(2)) + 0.5
            # v, compensate, reg_loss = self.prior_sampler.sample(bs=x.shape[0])
            # mask = mask * v.unsqueeze(2) * z
            # compensate = compensate.unsqueeze(2).repeat(1,1,2)
            # attention_probs = attention_probs * compensate

            # implementation 1:
            topk1 = torch.topk(not_null_probs, self.k,  dim=1)
            batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
            row_to_activate = batch_indices.repeat((1, self.k))
            mask[row_to_activate.view(-1), topk1.indices.view(-1), :] = 1
            v, compensate, reg_loss = self.prior_sampler.sample(
                self.log_alpha, self.log_beta, bs=x.shape[0])
            mask = mask * v.unsqueeze(2)
            compensate = compensate.unsqueeze(2).repeat(1, 1, 2)
            attention_probs = attention_probs * compensate

        # v, compensate, reg_loss = self.prior_sampler(bs=x.shape[0])
        # compensate = compensate.unsqueeze(2).repeat(1,1,2)
        # attention_probs = attention_probs / compensate

        mask = mask.squeeze()
        attention_probs = self.dropout(attention_probs)
        inputs = torch.matmul(attention_probs, value) * mask.unsqueeze(2)

        return inputs, mask, attention_probs[:, :, 0].detach(), reg_loss


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
        self.query = GroupLinearLayer(
            hidden_size, kdim * num_heads, num_blocks)
        self.value = GroupLinearLayer(
            hidden_size, hidden_size * num_heads, num_blocks)
        self.output_fc = GroupLinearLayer(
            num_heads * hidden_size, hidden_size, num_blocks)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, mask):
        key = self.key(h)
        query = self.query(h)
        value = self.value(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)
        value = self.transpose_for_scores(
            value, self.num_heads, self.hidden_size)

        scores = torch.matmul(query, key.transpose(-1, -2)
                              ) / math.sqrt(self.kdim)
        probs = nn.Softmax(dim=-1)(scores)

        mask = [mask for _ in range(probs.size(1))]
        mask = torch.stack(mask, dim=1)  # repeat activation mask for each head

        # inactive modules have zero-value query -> no context for them
        probs = probs * mask.unsqueeze(3)
        probs = self.dropout(probs)

        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context.size(
        )[:-2] + (self.num_heads * self.hidden_size,)
        # concatenate all heads
        context = context.view(*new_context_layer_shape)
        context = self.output_fc(context)  # to be add to current h

        return context


class Sparse_grad_attention(torch.autograd.Function):
    # def __init__(self, top_k):
    #     super(Sparse_grad_attention,self).__init__()
    #
    #     self.sa = Sparse_attention(top_k=top_k)

    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()


class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            #delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements
            #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps
            #delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
            delta = delta.reshape((delta.shape[0], 1))

        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        #print('attn', attn_w_normalize)

        return attn_w_normalize


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, topk, grad_sparse, attn_dropout=0.1, query_competition=False):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        self.query_compeition = query_competition
        self.softmax = nn.Softmax(dim=2)
        self.topk = topk
        self.grad_sparse = grad_sparse
        self.grad_sparse = grad_sparse
        #print('top 2 sparsity')
        self.topk = topk
        self.sa = Sparse_attention(top_k=topk)  # k=2
        #self.sga = Sparse_grad_attention(top_k=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        #print('in forward attn shape', attn.shape)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        #attn = self.dropout(attn)
        if not self.query_compeition:
            attn = self.softmax(attn)  # Shape: [N, num_q, num_k]
        else:
            # compete between queries
            attn = nn.Softmax(dim=1)(attn)
            attn = self.softmax(attn)  # Shape: [N, num_q, num_k]
            attn = attn + 1e-8  # to avoid unstability
            # compete between keys
            # Shape: [N, num_q, num_k]
            attn = attn / torch.sum(attn, dim=2, keepdim=True)
        # if random.uniform(0,1) < 0.0001 or attn[0].max() > 0.8:
        #    print('attn0', attn[0])

        #sparse_attn = attn*0.0
        #sparse_attn[:,0,0] += 1.0
        #sparse_attn[:,1,1] += 1.0
        #sparse_attn[:,2,2] += 1.0
        #attn = sparse_attn*1.0

        #extra_loss = 0.0
        # for k in range(0,3):
        #    extra_loss += 0.0001 * ((attn[:,k,k] - 1.0)**2).sum()
        extra_loss = 0.0

        use_sparse = True  # False

        if use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb*ins, outs))
            #print('sparse attn shape 1', sparse_attn.shape)
            #sga = Sparse_grad_attention(2)
            if self.grad_sparse:
                sga = Sparse_grad_attention(self.topk)
                sparse_attn = sga(sparse_attn)
            else:
                sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb, ins, outs))
            attn = sparse_attn*1.0

        output = torch.bmm(attn, v)
        return output, attn, extra_loss


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module 
    Args:
        `d_model_write`: size of vector with which to make keys/values
        `num_blocks_write`: number of vectors to use for keys/values
        `d_model_read`: size of vector with which to make queries
        `num_blocks_read`: number of vectors to use for queries
        `d_model_out`: size of output vector
    '''

    def __init__(self, n_head, d_model_read, d_model_write, d_model_out, d_k, d_v, num_blocks_read, num_blocks_write, topk, grad_sparse, n_templates, share_inp, share_comm, residual=True, dropout=0.1, skip_write=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        print('d model read', d_model_read)
        if share_inp:
            assert n_templates != 0, "provide number of paramters for sharing"
            self.GLN_qs = SharedGroupLinearLayer(
                d_model_read, n_head * d_k, n_templates)
            self.GLN_ks = GroupLinearLayer(
                d_model_write, n_head * d_k, num_blocks_write)
            self.GLN_vs = GroupLinearLayer(
                d_model_write, n_head * d_v, num_blocks_write)
        elif share_comm:
            # share Q,K,V for commuication
            assert n_templates != 0, "provide number of paramters for sharing"
            self.GLN_qs = SharedGroupLinearLayer(
                d_model_read, n_head * d_k, n_templates)
            self.GLN_ks = SharedGroupLinearLayer(
                d_model_write, n_head * d_k, n_templates)
            self.GLN_vs = SharedGroupLinearLayer(
                d_model_write, n_head * d_v, n_templates)
        else:
            self.GLN_qs = GroupLinearLayer(
                d_model_read, n_head * d_k, num_blocks_read)
            self.GLN_ks = GroupLinearLayer(
                d_model_write, n_head * d_k, num_blocks_write)
            self.GLN_vs = GroupLinearLayer(
                d_model_write, n_head * d_v, num_blocks_write)

        self.residual = residual

        #self.w_qs = nn.Linear(d_model_read, n_head * d_k)
        #self.w_ks = nn.Linear(d_model_write, n_head * d_k)
        #self.w_vs = nn.Linear(d_model_write, n_head * d_v)

        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5), topk=topk, grad_sparse=grad_sparse)
        #self.layer_norm = nn.LayerNorm(d_model)

        self.gate_fc = nn.Linear(n_head * d_v, d_model_out)

        if not skip_write:
            self.fc = nn.Linear(n_head * d_v, d_model_out)
        else:
            self.fc = lambda a: a

        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.ln = nn.LayerNorm(d_model_out)

    def forward(self, q, k, v, mask=None):

        #print('attn input shape', q.shape)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.GLN_qs(q).view(sz_b, len_q, n_head, d_k)
        #q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.GLN_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.GLN_vs(v).view(sz_b, len_v, n_head, d_v)
        #v = v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn, extra_loss = self.attention(q, k, v, mask=None)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        #print('output shape before fc', output.shape)

        # TODO: probably shouldn't just apply residual layer in the forward pass.

        output_init = output*1.0
        output = self.dropout(self.fc(output_init))
        gate = torch.sigmoid(self.gate_fc(output_init))

        #output = self.layer_norm(gate * output + (1 - gate) * residual)
        #output = gate * output + (1 - gate) * residual

        if self.residual:
            output = gate * torch.tanh(output)
        else:
            #output = self.ln(output)
            pass

        # output

        #print('attn', attn[0])
        #print('output input diff', output - residual)

        return output, attn, extra_loss


def main():
    x = torch.rand(2, 3, 4, requires_grad=False)
    mlp = nn.Linear(4, 4)
    x = mlp(x)
    argmax_x = ArgMax.apply(x)
    y = torch.matmul(argmax_x.float(), torch.randn(2, 4, 4))
    y = y.norm()
    y.backward()
    for p in mlp.parameters():
        print(p.grad)


if __name__ == "__main__":
    main()
