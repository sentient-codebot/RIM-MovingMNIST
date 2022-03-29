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
        num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon, input_size,
    ):
        """Build Slot Attention module.
        
        Args:
            num_iterations (int): number of iterations,
            num_slots (int): number of slots,
            slot_size (int): size of each slot,
            mlp_hidden_size (int): size of hidden layer in MLP,
            epsilon (float): epsilon for softmax,
            input_size (int): size of input,

        reference: https://github.com/google-research/google-research/blob/master/slot_attention/model.py
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.input_size = input_size

        self.norm_inputs = LayerNorm()
        self.norm_slots = LayerNorm()
        self.norm_mlp = LayerNorm()

        # Parameters for init (shared by all slots)
        self.slots_mu = torch.nn.parameter.Parameter(
            data = torch.randn(1, 1, self.slot_size),
        )
        self.slots_log_sigma = torch.nn.parameter.Parameter(
            data = torch.randn(1, 1, self.slot_size),
        )
        
        # Linear maps for attention module
        self.project_q = torch.nn.Linear(self.slot_size, self.slot_size, bias=False)
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

    def forward(self, inputs):
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
        slots = self.slots_mu + torch.exp(self.slots_log_sigma).to(inputs.device) * torch.randn(
            inputs.shape[0], self.num_slots, self.slot_size).to(inputs.device)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Compute attention scores.
            q = self.project_q(slots) # Shape: (batch_size, num_slots, slot_size).     
            attn_logits = torch.matmul(k, q.transpose(1, 2)) / math.sqrt(self.slot_size) # Shape: (batch_size, num_inputs, num_slots).
            attn = torch.softmax(attn_logits, dim=-1) # Shape: (batch_size, num_inputs, num_slots).

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=-2, keepdim=True) # NOTE what is the sum of `attn`?
            updates = torch.matmul(attn.transpose(1,2), v) # Shape: (batch_size, num_slots, slot_size).

            # Slots update.
            slots, _ = self.gru(
                updates.view(batch_size*self.num_slots, self.slot_size).unsqueeze(1), 
                slots.view(batch_size*self.num_slots, self.slot_size).unsqueeze(0)
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

def main():
    cudable = torch.cuda.is_available() 
    device = torch.device("cuda" if cudable else "cpu")
    slot_attn = SlotAttention(3, 4, 10, 20, 1e-8, 20).to(device)

    inputs = torch.randn(2, 5, 20)
    output = slot_attn(inputs)

    print(output.data)
    pass

if __name__ == "__main__":
    main()