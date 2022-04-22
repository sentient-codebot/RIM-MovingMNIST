from base64 import encode
from multiprocessing.sharedctypes import Value
from turtle import st
import torch
import torch.nn as nn
from rnn_models import RIMCell, RIM, SparseRIMCell, LayerNorm, Flatten, UnFlatten, Interpolate, SCOFFCell
from group_operations import GroupDropout
from collections import namedtuple
import numpy as np
from slot_attn.decoder_cnn import WrappedDecoder
from slot_attn.slot_attn import SlotAttention
from slot_attn.pos_embed import SoftPositionEmbed


Intm = namedtuple('IntermediateVariables',
    [
        'input_attn',
        'input_attn_mask',
        'blocked_dec',
    ])

class BasicEncoder(nn.Module):
    """basic encoder as baseline
    
    Args:
        `do_flatten`: whether to flatten the input
        `embedding_size`: size of the embedding
    
    Inputs:
        `x`: image of shape [N, 1, 64, 64]
    Output:
        `output`: output of the encoder; shape: [N, embedding_size]"""
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm() # Shape: (batch_size, 64, 6, 6)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64*6*6, embedding_size),  # Shape: [N, 6*6*64] -> [N, embedding_size]
            nn.ELU(),
            LayerNorm(),
        )
    def forward(self, x):
        x = self.conv_layer(x) # Shape: [N, 64, 6, 6]
        x = nn.Flatten(start_dim=1)(x) # Shape: [N, 64, 6, 6] -> [N, 64*6*6]
        x = self.mlp(x) # Shape: [N, 64*6*6] -> [N, embedding_size]
        return x.unsqueeze(1) # Shape: [N, 1, embedding_size]

class BasicDecoder(nn.Module):
    """Basic Upsampling Conv decoder that accepts concatenated hidden state vectors to decode an image
    
    Args:
        `embedding_size`: size of the embedding

    Inputs:
        `x`: hidden state of shape [N, embedding_size]

    Outputs:
        `output`: output of the decoder; shape: [N, 1, 64, 64]
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.layers = nn.Sequential(
            nn.Sigmoid(),
            LayerNorm(),
            nn.Linear(self.embedding_size, 4096), # Shape: [N, embedding_size] -> [N, 4096]
            nn.ReLU(),
            LayerNorm(),
            UnFlatten(), # Shape: [N, 4096] -> [N, 64, 8, 8]
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() # Shape; [N, 1, 64, 64]
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class NonFlattenEncoder(nn.Module):
    """nn.Module for slot attention encoder
    
    Args:
        `input_size`: size of input

    Inputs:
            `x`: image of shape [batch_size, 1, 64, 64]

    Outputs:
            `features`: feature vectors [batch_size, num_inputs, input_size]    
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm()
        )
        self.pos_emb = SoftPositionEmbed(64, (6, 6))
        self.mlp = nn.Sequential(
            nn.Linear(64, 64), # Shape: [batch_size, 6*6, 64]
            nn.ELU(),
            nn.Linear(64, self.input_size) # Shape: [batch_size, 6*6, input_size]
        )

    def forward(self, x):
        """
        Inputs:
            `x`: image of shape [batch_size, 1, 64, 64]

        Returns:
            `features`: feature vectors [batch_size, num_inputs, input_size]
        """
        x = self.cnn(x) # Shape: [batch_size, 64, 6, 6]
        x = self.pos_emb(x) # Shape: [batch_size, 64, 6, 6]
        x = x.permute(0, 2, 3, 1) # Shape: [batch_size, 6, 6, 64]
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[-1]) # Shape: [batch_size, 6*6, 64]
        x = self.mlp(x) # Shape: [batch_size, 6*6, input_size]

        return x

class BallModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_hidden = args.num_hidden # == num_rims

        self.slot_size = args.slot_size
        self.num_iterations_slot = args.num_iterations_slot
        self.num_slots = args.num_slots

        self.core = args.core.upper()
        self.use_sw = args.use_sw
        self.memory_size = args.memory_size
        self.num_memory_slots = args.num_memory_slots
        if self.use_sw:
            self.memory_mu = nn.parameter.Parameter(torch.randn(1, 1, self.memory_size)) # mean for initial memory. Shape: [1, 1, memory_size]
            self.memory_logvar = nn.parameter.Parameter(torch.randn(1, 1, self.memory_size)) # log variance for initial memory. Shape: [1, 1, memory_size]
        self.sparse = False
        self.get_intm = False
        self.use_slot_attention = args.use_slot_attention
        self.encoder_type = args.encoder_type
        self.decoder_type = args.decoder_type

        if self.encoder_type == "FLATTEN":
            self.encoder = BasicEncoder(embedding_size=self.input_size).to(self.args.device) # Shape: [batch_size, num_inputs, input_size]
            self.num_inputs = 1 # output of BasicFlattenEncoder
        elif self.encoder_type == "NONFLATTEN":
            self.encoder = NonFlattenEncoder(self.input_size).to(self.args.device) # Shape: [batch_size, num_inputs, input_size]
            self.num_inputs = 36 # output of NonFlattenEncoder
        else:
            raise ValueError("Invalid encoder type")

        self.slot_attention = None
        if self.use_slot_attention:
            self.slot_attention = SlotAttention(
                num_iterations=self.num_iterations_slot,
                num_slots=self.num_slots,
                slot_size=self.slot_size,
                mlp_hidden_size=128,
                epsilon=1e-8,
                input_size=self.input_size,
            ).to(self.args.device) # Shape: [batch_size,num_inputs, input_size] -> [batch_size, num_slots, slot_size]
            self.num_inputs = self.num_slots # number of output vectors of SlotAttention

        if args.decoder_type == "CAT_BASIC":
            self.decoder = BasicDecoder(embedding_size=self.num_hidden*self.hidden_size).to(self.args.device) # Shape: [batch_size, num_units*hidden_size] -> [batch_size, 1, 64, 64]
        elif args.decoder_type == "SEP_SBD":
            self.decoder = WrappedDecoder(self.hidden_size, decoder='transconv', mem_efficient=self.args.sbd_mem_efficient).to(self.args.device) # Shape: [batch_size, num_units, hidden_size] -> [batch_size, 1, 64, 64]
        else:
            raise NotImplementedError("Not implemented decoder type: {}".format(args.decoder_type))

        if self.core == 'RIM':
            if not self.sparse:
                self.rnn_model = RIMCell(
                                        device=self.args.device,
                                        input_size=self.slot_size if self.use_slot_attention else self.input_size, # NOTE: non-sensetive to num_inputs
                                        num_units=self.num_hidden,
                                        hidden_size=self.args.hidden_size,
                                        k=self.args.k,
                                        rnn_cell=self.args.rnn_cell, # defalt GRU
                                        input_key_size=self.args.input_key_size,
                                        input_value_size=self.args.input_value_size,
                                        num_input_heads = self.args.num_input_heads,
                                        input_dropout = self.args.input_dropout,
                                        use_sw = self.use_sw,
                                        comm_key_size = self.args.comm_key_size,
                                        comm_value_size = self.args.comm_value_size, 
                                        num_comm_heads = self.args.num_comm_heads, 
                                        comm_dropout = self.args.comm_dropout,
                                        memory_size = self.args.memory_size,
                ).to(self.args.device)
            else:
                raise NotImplementedError('Sparse RIM not updated with new args yet')
                self.rnn_model = SparseRIMCell(
                                        device=self.args.device,
                                        input_size=self.slot_size if self.slot_input else self.input_size, 
                                        num_units=self.args.num_units,
                                        hidden_size=self.args.hidden_size,
                                        k=self.args.k,
                                        rnn_cell='GRU', # defalt GRU
                                        input_key_size=self.args.input_key_size,
                                        input_value_size=self.args.input_value_size,
                                        input_query_size = self.args.input_query_size,
                                        num_input_heads = self.args.num_input_heads,
                                        input_dropout = self.args.input_dropout,
                                        comm_key_size = self.args.comm_key_size,
                                        comm_value_size = self.args.comm_value_size, 
                                        comm_query_size = self.args.comm_query_size, 
                                        num_comm_heads = self.args.num_comm_heads, 
                                        comm_dropout = self.args.comm_dropout,
                                        eta_0 = 2,
                                        nu_0 = 2,
                                        N = self.args.batch_size
                ).to(self.args.device)
        elif self.core == 'GRU':
            self.rnn_model = nn.GRU(
                                    input_size=self.slot_size*self.num_slots if self.use_slot_attention else self.input_size*self.num_inputs, # NOTE: sensetive to num_inputs
                                    hidden_size=self.args.hidden_size * self.args.num_units,
                                    num_layers=1,
                                    batch_first=True,
            ).to(self.args.device)
        elif self.core == 'LSTM':
            raise ValueError('LSTM Baseline Not Implemented Yet. ')
        elif self.core == "SCOFF":
            self.scoff_share_inp = True
            self.rnn_model = SCOFFCell(
                hidden_size=self.hidden_size*self.num_hidden,
                input_size=self.input_size if not self.use_slot_attention else self.slot_size,
                num_inputs=self.num_inputs if not self.use_slot_attention else self.num_slots,
                num_blocks_out=self.num_hidden,
                topkval=self.args.k,
                memorytopk=None, # not accessed
                step_att=True, # always do communication
                num_modules_read_input=None, # not accessed
                inp_heads=self.args.num_input_heads,
                comm_heads=self.args.num_comm_heads,
                do_gru=True if self.args.rnn_cell == 'GRU' else False,
                do_rel=False, # never True
                n_templates=self.args.num_rules,
                share_inp=self.scoff_share_inp, # all OFs share same input
                share_inp_attn=True,
                share_comm_attn=True,
            )
        else:
            raise ValueError('Illegal RNN Core')

    def forward(self, x, h_prev, M_prev=None):
        """
        Inputs:
            `x`: [batch_size, C, H, W]
            `h_prev`: [batch_size, num_units, hidden_size]`
            `c_prev`: (not implemented) [batch_size, num_units, ...]
            `M_prev`: (optional) [batch_size, num_units, memory_size] 
        Outputs:
            `dec_out`: [batch_size, 1, H, W]
            `h_out`: [batch_size, num_units, hidden_size]
            `c_out`: (not implemented) [batch_size, num_units, ...]
            `M_out`: [batch_size, num_units, memory_size] | None
            `Intm`: intermediate variables (for logging)
            """
        ctx = None
        M = None
        encoded_input = self.encoder(x) # Shape: (batch_size, 6*6, self.input_size) OR [batch_size, 1, self.input_size]
        if self.use_slot_attention:
            encoded_input = self.slot_attention(encoded_input) # Shape: [batch_size, num_slots, slot_size]

        reg_loss = 0.
        if self.core=='RIM':
            if not self.sparse:
                h_new, cs_new, M, ctx = self.rnn_model(x=encoded_input, hs=h_prev, cs=None, M=M_prev) 
            else:
                raise NotImplementedError('Sparse RIM not configured for slot input yet')
        elif self.core=='GRU':
            h_shape = h_prev.shape # Shape: [batch_size, num_units, hidden_size]
            h_prev = h_prev.reshape((h_shape[0],-1)) # flatten, Shape: [batch_size, num_units*hidden_size]
            _, h_new = self.rnn_model(encoded_input.flatten(start_dim=1).unsqueeze(1), # input shape: [N, 1, num_inputs*input_size|slot_size]
                                        h_prev.unsqueeze(0)) # h shape: [1, N, num_units*hidden_size]
            h_new = h_new.reshape(h_shape)
        elif self.core=='LSTM':
            raise NotImplementedError('LSTM core not implemented yet!')
        elif self.core=="SCOFF":
            # SCOFF requires a different input shape
            #   - input shape: [batch_size, num_object_files*input_size], we can pass different inputs to different OFs
            #   - h_prev shape: [batch_size, num_object_files*hidden_size]
            if not self.scoff_share_inp:
                encoded_input = encoded_input.view(encoded_input.shape[0], -1, 1).repeat(1, 1, self.num_hidden).flatten(start_dim=1) # Shape: [batch_size, num_object_files*input_size*num_hidden]
            h_prev = h_prev.view(h_prev.shape[0], -1) # Shape: [batch_size, num_units*hidden_size]
            h_new, c_new, mask, block_mask, temp_attention = self.rnn_model(inp=encoded_input, hx=h_prev, cx=None)
            h_new = h_new.view(h_new.shape[0], self.num_hidden, -1) # Shape: [batch_size, num_units, hidden_size]
        else:
            raise RuntimeError('Illegal RNN Core')
        
        blocked_out_ = torch.zeros(1).to(x.device)
        if "SEP" in self.decoder_type:
            dec_out_, channels, alpha_mask = self.decoder(h_new)
            if self.get_intm:
                blocked_out_ = channels*alpha_mask
        else:
            dec_out_ = self.decoder(h_new.view(h_new.shape[0],-1)) # Shape: [N, num_hidden*hidden_size] -> [batch_size, 1, 64, 64]

        if ctx is not None:
            intm = Intm(input_attn=ctx.input_attn, 
                input_attn_mask=ctx.input_attn_mask,
                blocked_dec=blocked_out_
                )
        else:
            intm = Intm(input_attn=torch.zeros(1), 
                input_attn_mask=torch.zeros(1),
                blocked_dec=blocked_out_
                )
        
        return dec_out_, h_new, M, intm

    def init_hidden(self, batch_size): 
        return torch.randn((batch_size, 
            self.args.num_hidden, 
            self.args.hidden_size), 
            requires_grad=False)
        
    def init_memory(self, batch_size):
        return torch.randn(
            (batch_size, self.num_memory_slots, self.memory_size),
            device=self.memory_mu.device
        )*torch.exp(self.memory_logvar) + self.memory_mu

    def nan_hook(self, out):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            raise RuntimeError(f"Found NAN in {self.__class__.__name__}: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def main():
    gamma = 0.1
    pass

class SpatialFlatten(nn.Module):
    def forward(self, input):
        """
        Inputs:
            `input`: a float tensor with shape [batch_size, C, H, W].
            
        Returns:
            `output`: a float tensor with shape [batch_size, H*W, C]."""
        output = input.permute(0, 2, 3, 1)
        output = output.contiguous()
        output = output.view(output.size(0), -1, output.size(3))

        return output

class SlotAttentionAutoEncoder(nn.Module):
    """AutoEncoder using SlotAttention for pretraining"""
    def __init__(self, input_size, num_iterations, num_slots, slot_size,):
        super().__init__()
        self.input_size = input_size
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.Encoder = NonFlattenEncoder(input_size=self.input_size) # output shape: [batch_size, num_inputs, input_size]
        self.slot_attention = SlotAttention(
                num_iterations=self.num_iterations,
                num_slots=self.num_slots,
                slot_size=self.slot_size,
                mlp_hidden_size=128,
                epsilon=1e-8,
                input_size=self.input_size,
            ) # output shape: [batch_size, num_slots, slot_size]
        self.Decoder = WrappedDecoder(hidden_size=self.slot_size, decoder='transconv') # input shape: [batch_size, num_slots, slot_size]

    def forward(self, x):
        """
        Inputs:
            `x`: a float tensor with shape [batch_size, C, H, W].
            
        Returns:
            `output`: a float tensor with shape [batch_size, C, H, W]."""
        encoded_input = self.Encoder(x)
        slot_attn = self.slot_attention(encoded_input)
        fused, channels, alpha_mask = self.Decoder(slot_attn)
        return fused

class TrafficModel(nn.Module):
    """Model for traffic4cast task"""
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        pass

if __name__ == "__main__":
    main()

