import torch
import torch.nn as nn
import math
from group_operations import GroupLinearLayer

from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial
from torch.distributions.uniform import Uniform

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

        
        out_probs = nn.Softmax(dim = -1)(attention_scores)[:,:, 0]

        return inputs, mask_, out_probs.detach(), torch.linalg.norm(out_probs, ord=1, dim=1).mean()

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
        switch_prior = switch_prior_sampler.rsample((bs,)).reshape(bs, self.num_blocks).to(self.device)
        # TODO compensate for expectation
        E_alpha = alpha/(alpha+beta).unsqueeze(0).repeat(bs, 1) # (1, num_blocks) * (bs, 1) -> (bs, num_blocks)
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
        lbeta_func = lambda alpha, beta: torch.lgamma(alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta)
        beta = torch.exp(log_beta)
        alpha = torch.exp(log_alpha)

        kl = lbeta_func(alpha, beta)-lbeta_func(self.alpha_0, self.beta_0) +\
            (self.alpha_0-alpha)*torch.digamma(self.alpha_0) +\
            (self.beta_0-beta)*torch.digamma(self.beta_0) +\
            (alpha+beta-self.alpha_0-self.beta_0)*torch.digamma(self.alpha_0+self.beta_0)

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
        approx_x, med= ctx.saved_tensors
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
        scaled_tanh = lambda x: torch.tanh(100*x)
        func_out, vjp = torch.autograd.functional.vjp(scaled_tanh, x, grad_output)
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
        self.query = GroupLinearLayer(hidden_size, kdim * num_heads, num_blocks)
        self.dropout = nn.Dropout(p = dropout)

        self.device = device

        self.eta_0 = torch.tensor(eta_0, device = device)
        self.nu_0 = torch.tensor(nu_0, device = device)
        self.beta_0 = self.nu_0+1
        self.alpha_0 = self.eta_0-self.nu_0+1
        self.num_blocks = num_blocks
        self.log_beta = nn.Parameter(torch.log(self.nu_0+1) + 0.1 * torch.randn(num_blocks, device=device))
        self.log_alpha = nn.Parameter(torch.log(self.eta_0-self.nu_0+1) + 0.1 * torch.randn(num_blocks, device=device))
        self.prior_sampler = PriorSampler(num_blocks, self.alpha_0, self.beta_0, device=device)

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
            topk1 = torch.topk(not_null_probs,self.k,  dim = 1)
            batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
            row_to_activate = batch_indices.repeat((1,self.k))
            mask[row_to_activate.view(-1), topk1.indices.view(-1), :] = 1
            v, compensate, reg_loss = self.prior_sampler.sample(self.log_alpha, self.log_beta, bs=x.shape[0])
            mask = mask * v.unsqueeze(2)
            compensate = compensate.unsqueeze(2).repeat(1,1,2)
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

class _sparse_max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        logits_dim = dim
        input = input.transpose(0, logits_dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        ctx_output = torch.max(torch.zeros_like(input), input - taus)
        ctx.save_for_backward(ctx_output)

        # Reshape back to original shape
        output = ctx_output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, logits_dim)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        ctx_output, = ctx.saved_tensors
        dim = 1
        nonzeros = torch.ne(ctx_output, 0)
        sum = torch.sum(grad_output.flatten(start_dim=0, end_dim=-2) * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        sum = sum.view(grad_output.shape[:-1])
        grad_input = nonzeros.reshape_as(grad_output) * (grad_output - sum.unsqueeze(-1).expand_as(grad_output))

        return grad_input, None


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=-1):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        output = _sparse_max.apply(input, self.dim)

        return output
