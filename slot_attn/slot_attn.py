import torch.nn as nn
import torch
import math

from collections import namedtuple
from .decoder_cnn import LayerNorm

# Ctx = namedtuple('RunningContext',
#     [
#         'input_attn',
#         'input_attn_mask'
#     ])

class SlotAttention(nn.Module):
    """Slot Attention module"""
    def __init__(self, 
        num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon, num_input, input_size,
        spotlight_bias=False,
        manual_init=False,
        manual_init_scale=None,
        ):
        """Build Slot Attention module.
        
        Args:
            num_iterations (int): number of iterations,
            num_slots (int): number of slots,
            slot_size (int): size of each slot,
            mlp_hidden_size (int): size of hidden layer in MLP,
            epsilon (float): epsilon for softmax,
            num_input (int): number of input,
            input_size (int): size of input,
            spotlight_bias (boolean): to decide whether you want to use spotlight or not 
            
            `manual_init`: whether you want to manually initialize slots
            `manual_init_scale`: [0, 1] or None, the scale of the manual initialization of slots. init_slots = scale*manual_init + (1-scale)*rand_init. If set to `None`, it will be learnable. 

        Inputs:
            `inputs`: (batch_size, num_inputs, input_size)
        
        Returns:
            `output`: (batch_size, num_slots, slot_size)

        reference: https://github.com/google-research/google-research/blob/master/slot_attention/model.py
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.input_size = input_size
        self.num_input= num_input
        self.spotlight_bias_loss=spotlight_bias
        self.norm_inputs = LayerNorm()
        self.norm_slots = LayerNorm()
        self.norm_mlp = LayerNorm()
        
        self.manual_init = manual_init
        self.manual_init_scale = manual_init_scale
        if self.manual_init:
            if self.manual_init_scale is not None:
                assert self.manual_init_scale >= 0 and self.manual_init_scale <= 1
            else:
                self.manual_init_scale_digit = nn.Parameter(torch.Tensor([0.])) # learnable, range [-inf, inf], need to be normalized

        # Parameters for init (shared by all slots)
        self.slots_mu = torch.nn.parameter.Parameter(
            data = torch.randn(1, 1, self.slot_size),
        )
        self.slots_log_sigma = torch.nn.parameter.Parameter(
            data = torch.randn(1, 1, self.slot_size),
        )
        
        # Linear maps for attention module
        self.project_q = torch.nn.Linear(self.slot_size, self.slot_size, bias=False)

        if self.spotlight_bias_loss:
            self.project_k = torch.nn.Linear(input_size, self.slot_size, bias=False)
            self.project_v = torch.nn.Linear(input_size, self.slot_size, bias=False)
            self.attn_param_bias = nn.Parameter(torch.randn(1, self.num_input, self.num_slots))
        else: 
            self.project_k = torch.nn.Linear(input_size, self.slot_size, bias=False)
            self.project_v = torch.nn.Linear(input_size, self.slot_size, bias=False)
        # Slot update functions
        self.gru = nn.GRU(input_size=self.slot_size, 
                            hidden_size=self.slot_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )
        
        self.do_logging = False
        self.hidden_features = {}
        self.hidden_features['attention_probs'] = []
        self.hidden_features['attention_map'] = []
        
    def manual_init_slots(self, manual_slots) -> torch.Tensor:
        """manually initialize slots
        
        Inputs:
            `slots`: past slots used for initialization. Shape [batch_size, num_slots, slot_size] **detached** tensor
            
        Returns:
            `initialized_slots`: initialized slots
        """
        rand_slots = self.slots_mu + torch.exp(self.slots_log_sigma).to(manual_slots.device) * torch.randn(
            manual_slots.shape[0], self.num_slots, self.slot_size).to(manual_slots.device)
        if manual_slots.requires_grad:
            print("Warning: slots are not detached. detaching now. ")
            manual_slots = manual_slots.detach()
        manual_init_scale = torch.sigmoid(self.manual_init_scale_digit)
        return manual_init_scale * manual_slots + (1.-manual_init_scale) * rand_slots

    def forward(self, inputs, init_slots=None):
        """
        Inputs:
            `inputs`: (batch_size, num_inputs, input_size)
        
        Returns:
            `output`: (batch_size, num_slots, slot_size)
        """
        batch_size = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs) # Shape: (batch_size, num_inputs, slot_size).
        v = self.project_v(inputs) # Shape: (batch_size, num_inputs, slot_size).

        # Initialize slots. Shape: (batch_size, num_slots, slot_size).
        if not self.manual_init or init_slots is None:
            slots = self.slots_mu + torch.exp(self.slots_log_sigma).to(inputs.device) * torch.randn(
                inputs.shape[0], self.num_slots, self.slot_size).to(inputs.device)
        else:
            slots = self.manual_init_slots(init_slots)

        # Multiple rounds of attention.
        self.hidden_features['attention_probs'] = []
        self.hidden_features['attention_map'] = []
        for iter_idx in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Compute attention scores.
            q = self.project_q(slots) # Shape: (batch_size, num_slots, slot_size).     
            attn_logits = torch.matmul(k, q.transpose(1, 2)) / math.sqrt(self.slot_size) # Shape: (batch_size, num_inputs, num_slots).
            if self.spotlight_bias_loss:
                attn_logits = attn_logits + self.attn_param_bias
            attn = torch.softmax(attn_logits, dim=-1) # Shape: (batch_size, num_inputs, num_slots).

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=-2, keepdim=True) # NOTE what is the sum of `attn`?
            updates = torch.matmul(attn.transpose(1,2), v) # Shape: (batch_size, num_slots, slot_size).
            if self.do_logging:
                with torch.no_grad():
                    bs, n_in, n_s = attn.shape
                    from math import sqrt
                    h, w = int(sqrt(n_in)), int(sqrt(n_in))
                    _attn_probs = attn.permute(0, 2, 1).view(bs, n_s, h, w).detach() # [batch_size, num_slots, h, w]
                    self.hidden_features['attention_probs'].append(_attn_probs)
                    self.hidden_features['attention_map'].append(
                        _attn_probs * inputs.view(bs, 1, h, w, -1).norm(dim=-1).detach() \
                        # [batch_size, num_slots, h, w]
                    )
                    if iter_idx == self.num_iterations - 1:
                        self.hidden_features['attention_probs'] = torch.stack(self.hidden_features['attention_probs'], dim=1) # [batch_size, *, num_slots, h, w]
                        self.hidden_features['attention_map'] = torch.stack(self.hidden_features['attention_map'], dim=1)

            # Slots update.
            slots, _ = self.gru(
                updates.view(batch_size*self.num_slots, self.slot_size).unsqueeze(1), 
                slots.view(batch_size*self.num_slots, self.slot_size).unsqueeze(0)
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        if self.spotlight_bias_loss:
            return slots, attn, self.attn_param_bias
        else:
            return slots

def main():
    cudable = torch.cuda.is_available() 
    device = torch.device("cuda" if cudable else "cpu")
    slot_attn = SlotAttention(3, 4, 10, 20, 1e-8, 20, True).to(device)

    inputs = torch.randn(2, 5, 20)
    output, attn, param_bias = slot_attn(inputs)

    print(output.data)
    print(attn)
    print(param_bias)
    pass

if __name__ == "__main__":
    main()