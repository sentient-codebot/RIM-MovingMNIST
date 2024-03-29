from tempfile import gettempdir
import torch
import torch.nn as nn
from rnn_models import RIMCell, RIM, SparseRIMCell, LayerNorm, Flatten, UnFlatten, Interpolate, SCOFFCell, AltSCOFFCell
from group_operations import GroupDropout, SharedGRUCell
from collections import namedtuple
import numpy as np
from slot_attn.decoder_cnn import WrappedDecoder, BroadcastConvDecoder
from slot_attn.slot_attn import SlotAttention
from slot_attn.pos_embed import SoftPositionEmbed
from utils import util
from utils.logging import enable_logging

class BasicEncoder(nn.Module):
    """basic encoder as baseline
    
    Args:
        `do_flatten`: whether to flatten the input
        `embedding_size`: size of the embedding
    
    Inputs:
        `x`: image of shape [N, in_channels=1|3, 64, 64]
    Output:
        `output`: output of the encoder; shape: [N, embedding_size]"""
    def __init__(self, embedding_size, in_channels=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.in_channels = in_channels
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=4, stride=2),
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
    def __init__(self, embedding_size, out_channels=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_channels = out_channels
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
            nn.Conv2d(16, self.out_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() # Shape; [N, 1, 64, 64]
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class BasicMLP(nn.Module):
    """Shared Basic MLP for compositional MLP
    
    Inputs:
        'x': hidden state portion of shape [N, hidden_size/factors_of_variations]  """
    def _init_(self, hidden_size, fov):
        super()._init_()
        self.hidden_size = hidden_size
        self.fov = fov
        self.mlp =  nn.Sequential(
                            nn.Linear(int(self.hidden_size/self.fov), 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, self.hidden_size),
                        )
    def forward(self, x):
        x = self.mlp(x)
        return x
    


class SharedBasicDecoder(nn.Module):
    """Shared Basic decoder for group of latents
    
    Inputs:
        `x`: hidden state of shape [N, M, embedding_size]"""
    def __init__(self, embedding_size, out_channels, cell_switch=(), norm_method=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_channels = out_channels # 1 or 3
        self.cnn = BasicDecoder(embedding_size, out_channels+1)
        self.cell_switch = cell_switch
        self.norm_method = norm_method if norm_method is not None else 'default'
        
    def forward(self, x):
        x = x.permute(1, 0, 2) # Shape: [M, N, embedding_size] -> [M, N, embedding_size]
        out_list = []
        mask_list = []
        for component in x: # Shape: [N, embedding_size]
            out_tensor = self.cnn(component) # [N, C+1, H, W]
            out_list.append(out_tensor[:, :-1, :, :])
            mask_list.append(out_tensor[:, -1:, :, :])
        channels = torch.stack(out_list, dim=1) # Shape: [N, M, C, H, W]
        mask = torch.stack(mask_list, dim=1) # Shape: [N, M, 1, H, W]
        for cell_idx in self.cell_switch:
            mask[:, cell_idx, :, :, :] = float('-inf')
        mask = mask/torch.sum(mask, dim=1, keepdim=True) # Normalization. Shape: [N, M, 1, H, W]
        
        fused = torch.sum(channels*mask, dim=1) # Shape: [N, C, H, W]
        return fused, channels, mask


class Compositional_MLP(nn.Module):
    """Shared Basic MLP for hidden variables
    
    Inputs:
        `x`: hidden state of shape [N, M, embedding_size]"""
    def __init__(self, embedding_size, fov):
        super().__init__()
        self.fov = fov
        self.embedding_size = embedding_size
        self.mlp = BasicMLP(embedding_size, fov)
        
    def forward(self, x):

        x = x.permute(1, 0, 2) # Shape: [N, M, embedding_size] -> [M, N, embedding_size]
        out_list = []
        mask_list = []
        slots_list = []
        for component in x: # Shape: [N, embedding_size]
            chuncks = torch.chunck(component, self.fov, dim=1) # Shape: [fov, N, embedding_size/fov]
            for fov in range(self.fov):
                out_latent = self.mlp(chuncks[fov]) # [N, embedding_size]
                mask_latent = self.mlp(chuncks[fov])  # [N, embedding_size]
                out_list.append(out_latent)
                mask_list.append(mask_latent)
            channels = torch.stack(out_list, dim=1) # Shape: [N, fov, embedding_size]
            mask = torch.stack(mask_list, dim=1) # Shape: [N, fov, embedding_size]
            mask = mask/torch.sum(mask, dim=1, keepdim=True) # Normalization.  Shape: [N, fov, embedding_size]
            fused = torch.sum(channels*mask, dim=1) # Shape: [N, embedding_size]
            slots_list.append(fused)
        latent_variables = torch.stack(slots_list, dim=1)

        return latent_variables
    
class SharedBroadcastDecoder(nn.Module):
    """
    """
    def __init__(self, embedding_size, out_channels):
        super().__init__()
        self.embedding_size = embedding_size
        self.out_channels = out_channels # 1 or 3
        self.cnn = BroadcastConvDecoder(embedding_size,) # fixed output dimension (3, 64, 64)
        
    def forward(self, x):
        x = x.permute(1, 0, 2) # Shape: [N, M, embedding_size] -> [M, N, embedding_size]
        out_list = []
        mask_list = []
        for component in x: # Shape: [N, embedding_size]
            out_tensor, out_mask = self.cnn(component) # [N, C+1, H, W]
            out_list.append(out_tensor)
            mask_list.append(out_mask)
        channels = torch.stack(out_list, dim=1) # Shape: [N, M, C, H, W]
        # mask = torch.stack(mask_list, dim=1) # Shape: [N, M, 1, H, W]
        # mask = nn.Softmax(dim=1)(mask) # Shape: [N, M, 1, H, W]
        mask = torch.stack(mask_list, dim=1) # Shape: [N, M, 1, H, W]
        mask = mask/(torch.sum(mask, dim=1, keepdim=True)+1e-8) # Normalization. Shape: [N, M, 1, H, W]
        fused = torch.sum(channels*mask, dim=1) # Shape: [N, C, H, W]
        return fused, channels, mask

class NonFlattenEncoder(nn.Module):
    """nn.Module for slot attention encoder
    
    Args:
        `input_size`: size of input

    Inputs:
            `x`: image of shape [batch_size, in_channels=1|3, 64, 64]

    Outputs:
            `features`: feature vectors [batch_size, num_inputs, input_size]    
    """
    def __init__(self, input_size, in_channels=1):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=4, stride=2),
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

class SynMOTEncoder(nn.Module):
    """nn.Module for slot attention encoder
    
    Args:
        `input_size`: size of input

    Inputs:
            `x`: image of shape [batch_size, 3, 64, 64]

    Outputs:
            `features`: feature vectors [batch_size, num_inputs, input_size]    
    """
    def __init__(self, input_size, do_flatten = False):
        super().__init__()
        self.input_size = input_size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm() # [N, 36, 128]
        )
        self.do_flatten = do_flatten
        if self.do_flatten:
            self.mlp = nn.Sequential(
                nn.Linear(128*6*6, input_size),
                nn.ELU(),
                LayerNorm(),
            )
        else:
            self.pos_emb = SoftPositionEmbed(128, (6, 6))
            self.mlp = nn.Sequential(
                nn.Linear(128, 256), # Shape: [batch_size, 6*6, 64]
                nn.ELU(),
                nn.Linear(256, self.input_size) # Shape: [batch_size, 6*6, input_size]
            )

    def forward(self, x):
        """
        Inputs:
            `x`: image of shape [batch_size, 3, 64, 64]

        Returns:
            `features`: feature vectors [batch_size, num_inputs, input_size]
        """
        x = self.cnn(x) # Shape: [batch_size, 64, 6, 6]
        if self.do_flatten:
            x = self.mlp(x.flatten(start_dim=1))
            x = x.unsqueeze(1) # Shape: [batch_size, 1, input_size]
        else:
            x = self.pos_emb(x) # Shape: [batch_size, 64, 6, 6]
            x = x.permute(0, 2, 3, 1) # Shape: [batch_size, 6, 6, 64]
            x = x.contiguous()
            x = x.view(x.shape[0], -1, x.shape[-1]) # Shape: [batch_size, 6*6, 64]
            x = self.mlp(x) # Shape: [batch_size, 6*6, input_size]

        return x
    
    def output_shape(self, input_size: tuple[int]):
        out = input_size
        for cnn_layer in self.cnn:
            if not isinstance(cnn_layer, nn.Conv2d):
                continue
            out_channels, kernel_size, stride, padding, dilation \
                = cnn_layer.out_channels, cnn_layer.kernel_size, cnn_layer.stride, cnn_layer.padding, cnn_layer.dilation
            out = conv2d_output_size(
                out, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        return out
class SpritesMOTEncoder(nn.Module):
    """nn.Module for slot attention encoder
    
    Args:
        `input_size`: size of input

    Inputs:
            `x`: image of shape [batch_size, 3, 128, 128]

    Outputs:
            `features`: feature vectors [batch_size, num_inputs, input_size]    
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            LayerNorm()
        )
        self.pos_emb = SoftPositionEmbed(128, (6, 6))
        self.mlp = nn.Sequential(
            nn.Linear(128, 128), # Shape: [batch_size, 6*6, 64]
            nn.ELU(),
            nn.Linear(128, self.input_size) # Shape: [batch_size, 6*6, input_size]
        )

    def forward(self, x):
        """
        Inputs:
            `x`: image of shape [batch_size, 3, 128, 128]

        Returns:
            `features`: feature vectors [batch_size, num_inputs = 36, input_size]
        """
        x = self.cnn(x) # Shape: [batch_size, 64, , ]
        # x = self.pos_emb(x) # Shape: [batch_size, 64, , ]
        x = x.permute(0, 2, 3, 1) # Shape: [batch_size, , , 64]
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[-1]) # Shape: [batch_size, 6*6, 64]
        x = self.mlp(x) # Shape: [batch_size, , input_size]

        return x
    
    def output_shape(self, input_size: tuple[int]):
        out = input_size
        for cnn_layer in self.cnn:
            if not isinstance(cnn_layer, nn.Conv2d):
                continue
            out_channels, kernel_size, stride, padding, dilation \
                = cnn_layer.out_channels, cnn_layer.kernel_size, cnn_layer.stride, cnn_layer.padding, cnn_layer.dilation
            out = conv2d_output_size(
                out, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        return out

class BallModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_hidden = args.num_hidden # == num_rims
        self.spotlight_bias=args.spotlight_bias
        self.slot_size = args.slot_size
        self.num_iterations_slot = args.num_iterations_slot
        self.num_slots = args.num_slots
        self.bs=args.batch_size
        self.core = args.core.upper()
        self.use_sw = args.use_sw
        self.memory_size = args.memory_size
        self.num_memory_slots = args.num_memory_slots
        if self.use_sw:
            self.memory_mu = nn.parameter.Parameter(torch.randn(1, 1, self.memory_size)) # mean for initial memory. Shape: [1, 1, memory_size]
            self.memory_logvar = nn.parameter.Parameter(torch.randn(1, 1, self.memory_size)) # log variance for initial memory. Shape: [1, 1, memory_size]
        self.sparse = False
        self.use_slot_attention = args.use_slot_attention
        self.encoder_type = args.encoder_type
        self.decoder_type = args.decoder_type
        self.use_compositional_MLP = args.use_compositional_MLP
        
        self.cell_switch = args.cell_switch
        

        if self.args.task in ['SPRITESMOT', 'VMDS', 'VOR', 'MSPRITES']:
            self.encoder = SynMOTEncoder(
                input_size=self.input_size,
                do_flatten=True if self.encoder_type == "FLATTEN" else False,
            )
            self.num_inputs=1 if self.encoder_type == "FLATTEN" else 36
            self.fov = 2 # factor of variations
        else:
            self.fov=2
            if self.encoder_type == "FLATTEN":
                self.encoder = BasicEncoder(embedding_size=self.input_size).to(self.args.device) # Shape: [batch_size, num_inputs, input_size]
                self.num_inputs = 1 # output of BasicFlattenEncoder
            elif self.encoder_type == "NONFLATTEN":
                self.encoder = NonFlattenEncoder(self.input_size).to(self.args.device) # Shape: [batch_size, num_inputs, input_size]
                self.num_inputs = 36 # output of NonFlattenEncoder
            else:
                raise ValueError("Invalid encoder type")
        self.resolution = [int(np.sqrt(self.num_inputs)),int(np.sqrt(self.num_inputs))]
        self.slot_attention = None
        if self.use_slot_attention:
            self.slot_attention = SlotAttention(
                num_iterations=self.num_iterations_slot,
                num_slots=self.num_slots,
                slot_size=self.slot_size,
                mlp_hidden_size=128,
                epsilon=1e-8,
                num_input=self.num_inputs,
                input_size=self.input_size,
                spotlight_bias=self.spotlight_bias,
                manual_init=self.args.use_past_slots,
                eval_softmax_temp=self.args.sa_eval_sm_temp,
                do_key_norm=self.args.sa_key_norm,
            ).to(self.args.device) # Shape: [batch_size,num_inputs, input_size] -> [batch_size, num_slots, slot_size]
            self.num_inputs = self.num_slots # number of output vectors of SlotAttention
        if self.use_compositional_MLP:
            self.compositional_MLP = Compositional_MLP(self.slot_size, self.fov)
        self.decode_hidden = args.decode_hidden
        if not self.decode_hidden:
            self.embedding_size = self.slot_size if self.use_slot_attention else self.input_size # embedding size for the decoder
            if self.encoder_type == 'NONFALTTEN' and 'CAT' in self.decoder_type:
                print("Warning: NONFLATTEN + CAT: weird setting. avoid. ")
                self.embedding_size = self.input_size * self.num_inputs
        else:
            self.embedding_size = self.hidden_size
            if 'CAT' in self.decoder_type:
                self.embedding_size = self.hidden_size * self.num_hidden

        out_channels = 1
        _sbd_decoder = 'transconv'
        if self.args.task in ['SPRITESMOT', 'VMDS', 'VOR', 'MSPRITES']:
            out_channels = 3
            _sbd_decoder = 'synmot'
        if args.decoder_type == "CAT_BASIC":
            self.decoder = BasicDecoder(embedding_size=self.embedding_size, out_channels=out_channels) # Shape: [batch_size, num_units*hidden_size] -> [batch_size, 1, 64, 64]
        elif args.decoder_type == "SEP_BASIC":
            self.decoder = SharedBasicDecoder(embedding_size=self.embedding_size, 
                                              out_channels=out_channels, 
                                              cell_switch=self.cell_switch,
                                              norm_method=self.args.dec_norm_method)
        elif args.decoder_type == "SEP_SBD":
            if self.args.task in ['SPRITESMOT', 'VMDS', 'VOR', 'MSPRITES'] and False:
                self.decoder = SharedBroadcastDecoder(self.embedding_size, 3)
            else:
                self.decoder = WrappedDecoder(self.embedding_size, 
                                              decoder=_sbd_decoder, 
                                              mem_efficient=self.args.sbd_mem_efficient, 
                                              cell_switch=self.cell_switch,
                                              norm_method=self.args.dec_norm_method) # Shape: [batch_size, num_units, hidden_size] -> [batch_size, 1, 64, 64]
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
                                        use_rule_sharing=self.args.use_rule_sharing,
                                        use_rule_embedding=self.args.use_rule_embedding,
                                        num_rules=self.args.num_rules,
                                        hard_input_attention=self.args.hard_input_attention,
                                        null_input_type=self.args.null_input_type,
                                        input_attention_key_norm=self.args.input_attention_key_norm,
                                        cell_switch=self.cell_switch,
                )
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
            # self.rnn_model = RIMCell(
            #     device=self.args.device,
            #     input_size=self.slot_size*self.num_slots if self.use_slot_attention else self.input_size*self.num_inputs, # NOTE: sensetive to num_inputs
            #     num_units=1, # NOTE one bulky GRU
            #     hidden_size=self.hidden_size * self.num_hidden,
            #     k=1,
            #     rnn_cell='GRU', # defalt GRU
            #     input_key_size=self.args.input_key_size,
            #     input_value_size=self.args.input_value_size,
            #     num_input_heads = self.args.num_input_heads,
            #     input_dropout = 0.,
            #     use_sw = False,
            #     comm_key_size = self.args.comm_key_size,
            #     comm_value_size = self.args.comm_value_size, 
            #     num_comm_heads = self.args.num_comm_heads, 
            #     comm_dropout = 0.,
            #     memory_size = self.args.memory_size,
            #     use_rule_sharing=self.args.use_rule_sharing,
            #     use_rule_embedding=self.args.use_rule_embedding,
            #     num_rules=self.args.num_rules,
            # )
            self.rnn_model = nn.GRUCell(
                                    input_size=self.slot_size*self.num_slots if self.use_slot_attention else self.input_size*self.num_inputs, # NOTE: sensetive to num_inputs
                                    hidden_size=self.args.hidden_size * self.args.num_hidden,
            )
            self.rnn_model.hidden_features = {}
        elif self.core == 'LSTM':
            self.rnn_model = RIMCell(
                device=self.args.device,
                input_size=self.slot_size*self.num_slots if self.use_slot_attention else self.input_size*self.num_inputs, # NOTE: sensetive to num_inputs
                num_units=1, # NOTE one bulky GRU
                hidden_size=self.hidden_size * self.num_hidden,
                k=1,
                rnn_cell='LSTM', # defalt GRU
                input_key_size=self.args.input_key_size,
                input_value_size=self.args.input_value_size,
                num_input_heads = self.args.num_input_heads,
                input_dropout = 0.,
                use_sw = False,
                comm_key_size = self.args.comm_key_size,
                comm_value_size = self.args.comm_value_size, 
                num_comm_heads = self.args.num_comm_heads, 
                comm_dropout = 0.,
                memory_size = self.args.memory_size,
                use_rule_sharing=self.args.use_rule_sharing,
                use_rule_embedding=self.args.use_rule_embedding,
                num_rules=self.args.num_rules,
            )
        elif self.core == "SCOFF":
            self.scoff_share_inp = True
            self.rnn_model = SCOFFCell(
                hidden_size=self.hidden_size*self.num_hidden,
                input_size=self.input_size if not self.use_slot_attention else self.slot_size,
                num_inputs=self.num_inputs if not self.use_slot_attention else self.num_slots,
                num_hidden=self.num_hidden,
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
                straight_through_input=self.args.slot_straight_input,
                hard_input_attention=self.args.hard_input_attention,
                null_input_type=self.args.null_input_type,
            )
        elif self.core == "ALTSCOFF":
            self.rnn_model = AltSCOFFCell(
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
                use_rule_embedding=self.args.use_rule_embedding,
                num_rules=self.args.num_rules,
                hard_input_attention=self.args.hard_input_attention,
                null_input_type=self.args.null_input_type,
                input_attention_key_norm=self.args.input_attention_key_norm,
            )
        else:
            raise ValueError('Illegal RNN Core')
        
        self.latent_transform = None
        if not self.decode_hidden:
            if 'CAT' in args.decoder_type:
                self.latent_transform = nn.Sequential(
                    nn.Linear(self.num_hidden*self.hidden_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.embedding_size)
                ) # hidden_state -> image embedding
            else:
                self.latent_transform = nn.Sequential(
                    nn.Linear(self.hidden_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.embedding_size)
                )
        
        self.past_slots = None
        
        self.do_logging = False
        self.hidden_features = {}

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
        # default values of intermediate variables
        ctx = None
        rules_selected = torch.zeros(1).to(x.device)

        # computations
        M = None
        with torch.no_grad() if self.args.load_trained_slot_attention else torch.enable_grad():
            encoded_input = self.encoder(x) # Shape: (batch_size, 6*6, self.input_size) OR [batch_size, 1, self.input_size]
        if self.use_slot_attention:
            with torch.no_grad() if self.args.load_trained_slot_attention else torch.enable_grad():
                if self.spotlight_bias:
                    #__u = lambda x: util.unpack_seqdim(x, self.bs)
                    encoded_input, attn, attn_param_bias = self.slot_attention(encoded_input, self.past_slots) # Shape: [batch_size, num_slots, slot_size]
                    grid_val = util.build_grid2D(self.resolution).repeat(encoded_input.shape[0],1,1,1).reshape([encoded_input.shape[0],-1,2]).to(h_prev.device)
                    slot_means = torch.matmul(attn.permute(0,2,1),grid_val)
                    slot_means_ = slot_means.unsqueeze(1) # [N, 1, num_slots, 2]
                    grid_val_ = grid_val.unsqueeze(2) # [N, num_inputs, 1, 2]
                    slot_variances_ = ((slot_means_ - grid_val_)**2).sum(-1)
                    slot_variances_ = slot_variances_ * attn
                    slot_variances = slot_variances_.sum(-1)
                
                else:
                    encoded_input = self.slot_attention(encoded_input, self.past_slots) # Shape: [batch_size, num_slots, slot_size]
                    if self.do_logging:
                        self.hidden_features['slots'] = encoded_input # for logging. [N, num_slots, slot_size]
            

            if self.slot_attention.manual_init:
                self.past_slots = encoded_input.detach()
            else:
                self.past_slots = None

        if self.use_compositional_MLP:
            encoded_input = self.compositional_MLP(encoded_input)

                
                

        reg_loss = 0.
        # UPDATE
        # pass through self.rnn_model (RNN core)
        if self.core=='RIM' or self.core=='ALTSCOFF':
            if not self.sparse:
                h_new, cs_new, M = self.rnn_model(x=encoded_input, hs=h_prev, cs=None, M=M_prev) # one-step prediciton
            else:
                raise NotImplementedError('Sparse RIM not configured for slot input yet')
        elif self.core=='GRU':
            h_shape = h_prev.shape # Shape: [batch_size, num_units, hidden_size]
            gru_input = encoded_input.view(encoded_input.shape[0], -1)
            gru_hidden = h_prev.view(h_prev.shape[0], -1)
            
            h_new = self.rnn_model(gru_input, gru_hidden)
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

            # for logging
            if self.do_logging:
                rules_selected = torch.argmax(temp_attention, dim=2, keepdim=False) # temp_attention: [bs, num_hidden, n_templates] -> LongTensor [bs, num_hidden]
                self.rnn_model.hidden_features['rule_attn_argmax'] = rules_selected
                self.rnn_model.hidden_features['rule_attn_probs'] = temp_attention
        else:
            raise RuntimeError('Illegal RNN Core')
        
        
        # DECODING
        # newly added, transform h_new back to slots
        # self.hidden_features['individual_output'] = torch.empty((h_new.shape[0],self.num_hidden,1,64,64)).to(x.device)
        # self.hidden_features['individual_output_unmasked'] = torch.empty((h_new.shape[0],self.num_hidden,1,64,64)).to(x.device)
        curr_dec_out_ = None
        curr_alpha_mask = None
        object_mask = None
        if not self.decode_hidden:
            if 'CAT' in self.decoder_type:
                pred_latent = self.latent_transform(h_new.view(h_new.shape[0],-1)) # Shape: [N, K*hidden_size] -> [N, embedding_size]
            else:
                pred_latent = self.latent_transform(h_new) # Shape: [N, K, hidden_size] -> [N, K, slot/input_size]
            
            if "SEP" in self.decoder_type:
                curr_dec_out_, curr_channels, curr_alpha_mask = self.decoder(encoded_input)
                next_dec_out_, next_channels, next_alpha_mask = self.decoder(pred_latent)
                if self.do_logging or True: # always log ind_output
                    blocked_out_ = next_channels*next_alpha_mask
                    self.hidden_features['individual_output'] = blocked_out_.detach()
                    self.hidden_features['individual_output_unmasked'] = next_channels.detach()
                    self.hidden_features['individual_recons'] = (curr_channels*curr_alpha_mask).detach()
            else:
                curr_dec_out_ = self.decoder(encoded_input.view(encoded_input.shape[0],-1)) # Shape: [N, num_hidden*hidden_size] -> [batch_size, 1, 64, 64]
                next_dec_out_ = self.decoder(pred_latent.view(pred_latent.shape[0],-1)) # Shape: [N, num_hidden*hidden_size] -> [batch_size, 1, 64, 64]
        else:
            if "SEP" in self.decoder_type:
                next_dec_out_, next_channels, next_alpha_mask = self.decoder(h_new)
                if self.do_logging or True: # always log ind_output
                    blocked_out_ = next_channels*next_alpha_mask
                    self.hidden_features['individual_output'] = blocked_out_.detach()
                    self.hidden_features['individual_output_unmasked'] = next_channels.detach()
            else:
                next_dec_out_ = self.decoder(h_new.view(h_new.shape[0],-1)) # Shape: [N, num_hidden*hidden_size] -> [batch_size, 1, 64, 64]
        
        if getattr(self, 'mot_eval', False):
            """output evaluation stats for MOT"""
            object_mask = self.object_mask(curr_channels, curr_alpha_mask)
            if self.spotlight_bias:
                return curr_dec_out_, next_dec_out_, h_new, M, slot_means, slot_variances, attn_param_bias, object_mask
            else:
                return curr_dec_out_, next_dec_out_, h_new, M, object_mask # # (BS, <K>, 1, H, W)
        
        if self.spotlight_bias:
            return curr_dec_out_, next_dec_out_, h_new, M, slot_means, slot_variances, attn_param_bias
        else:
            return curr_dec_out_, next_dec_out_, h_new, M

    def object_mask(self, channels, alpha_mask):
        """
        channels: [N, k, C, H, W]
        alpha_mask: [N, k, 1, H, W]
        """
        gray_scaled = torch.norm(channels, dim=2, keepdim=True) # [N, k, 1, H, W]
        object_mask = gray_scaled * alpha_mask # [N, k, 1, H, W]
        return object_mask

    def init_hidden(self, batch_size): 
        """init hidden states, also clear out past slots"""
        if self.past_slots is not None:
            self.past_slots = None
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
        self.encoder = NonFlattenEncoder(input_size=self.input_size) # output shape: [batch_size, num_inputs, input_size]
        self.slot_attention = SlotAttention(
                num_iterations=self.num_iterations,
                num_slots=self.num_slots,
                slot_size=self.slot_size,
                mlp_hidden_size=128,
                epsilon=1e-8,
                input_size=self.input_size,
            ) # output shape: [batch_size, num_slots, slot_size]
        self.decoder = WrappedDecoder(hidden_size=self.slot_size, decoder='transconv') # input shape: [batch_size, num_slots, slot_size]

    def forward(self, x):
        """
        Inputs:
            `x`: a float tensor with shape [batch_size, C, H, W].
            
        Returns:
            `output`: a float tensor with shape [batch_size, C, H, W]."""
        encoded_input = self.encoder(x)
        slot_attn = self.slot_attention(encoded_input)
        fused, channels, alpha_mask = self.decoder(slot_attn)
        return fused

class TrafficModel(nn.Module):
    """Model for traffic4cast task"""
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        pass
    
def conv2d_output_size(
    input_size: tuple[int],
    out_channels: int,
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
) -> tuple[int]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    n, c_in, h_in, w_in = input_size
    c_out = out_channels
    h_out = (h_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
    w_out = (w_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
    
    return (n, c_out, h_out, w_out)
        
        


if __name__ == "__main__":
    main()

