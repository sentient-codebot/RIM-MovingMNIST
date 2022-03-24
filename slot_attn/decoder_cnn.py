from turtle import back
from numpy import broadcast
import torch
import torch.nn as nn
from pos_embed import SoftPositionEmbed

class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

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

def make_decoder():
    """make decoder
    decoder same as slot decoder. 
        using transposedConv2d 
        not using interpolate+conv2d
    
    `input`: size (BS*num_blocks, C==hidden_size, init_H, init_W)"""

    # H out​ =(H in​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    return nn.Sequential(
        nn.ConvTranspose2d(100, 64, 5, 2, padding=2, output_padding=1), # 8*2 + 3 - 4 + 1 = 16
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, padding=2, output_padding=1), # 16*2 + 3 - 4 + 1 = 32
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 3, 2, padding=1, output_padding=1), # 32*2 +1 -2 +1 = 64
        nn.ReLU(),
        nn.ConvTranspose2d(64, 2, 3, 1, padding=1), # 64 + 2 - 2 = 64
    )

def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask.
    
    `input`: (bs*num_vecs, num_channels+1, h, w) """
    height, width = x.shape[2:]
    unstacked = torch.reshape(x, (batch_size, x.shape[0]//batch_size, -1, height, width)) # (bs, num_vecs, num_channel+1, h, w)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2) # (bs, num_vecs, *, height, width)
    return channels, masks

def spatial_broadcast(slots, resolution):
    """broadcast K x slots into everywhere in a grid
    
    Input:
        `slots`: (BS, K, d_slot)
        `resolution`: (H, W) the target image size to broadcast to
        
    Return:
        `img`: (BS*K, d_slot, H, W) """

    slots = slots.reshape(slots.shape[0]*slots.shape[1], -1, 1, 1) # (BS*K, d_slot, 1, 1)
    img = slots.repeat((1, 1, *resolution))

    return img

class WrappedDecoder(nn.Module):
    """Decoder that takes slots embeddings and perform decoding individually and combine. 
    
    Input:
        `hidden`: (BS, K, d_slot) """
    def __init__(self, hidden_size):
        super().__init__()
        self.decoder = make_decoder()
        self.pos_embed = SoftPositionEmbed(hidden_size, (8,8))

    def forward(self, hidden):
        broadcast_hidden = spatial_broadcast(hidden, (8,8)) # (BS*K, d_slot, 8, 8)
        broadcast_hidden = self.pos_embed(broadcast_hidden)
        dec_out = self.decoder(broadcast_hidden) # (BS*K, 2, 64, 64)
        channels, alpha_mask = unstack_and_split(dec_out, batch_size=hidden.shape[0], num_channels=1) # (BS, K, *, H, W)
        alpha_mask = torch.nn.Softmax(dim=1)(alpha_mask) # (BS, <K>, 1, H, W)
        masked_channels = channels*alpha_mask
        fused = torch.sum(masked_channels, dim=1)

        return fused


if __name__ == '__main__':
    decoder = make_decoder()

    # broadcast_hidden = torch.randn(2*6, 100, 8, 8)
    # hidden = torch.arange(2*6*100).reshape(2, 6, 100)
    hidden = torch.randn(2, 6, 100)
    broadcast_hidden = spatial_broadcast(hidden, (8, 8))

    out = decoder(broadcast_hidden)
    print(out.shape)

    c, m = unstack_and_split(out, batch_size=2, num_channels=1) # (N, K, *, H, W)
    m = torch.nn.Softmax(dim=1)(m) # (N, <K>, 1, H, W)
    masked_c = c*m # first ele-wise prod to apply mask (equally) for every RGB channel
    # masked_c: (N, K, 3/1, H, W)
    fused = torch.sum(masked_c, dim=1) # sum over K dim (slot dimension)

    print(fused.shape)

    pass