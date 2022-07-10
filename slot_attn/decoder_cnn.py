from turtle import back
from numpy import broadcast
import torch
import torch.nn as nn
from .pos_embed import SoftPositionEmbed

class NormReLU(nn.Module):
    """function similar to Softmax, first activate with relu, then normalize by their sum

    """
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.dim = dim
    
    def forward(self, x):
        """forward function of NormReLU

        Arguments:
            x -- tensor of dimension (*) >= self.dim
        """
        x = self.relu(x)
        x_sum = torch.sum(x, dim=self.dim, keepdim=True)
        x = x/(1e-8+x_sum)
        
        return x

class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

# class UnFlatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), 64, 8, 8)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

def make_decoder_transconv(hidden_size):
    """make decoder
    decoder same as slot decoder. 
        using transposedConv2d 
        not using interpolate+conv2d
    
    `input`: size (BS*num_blocks, C=hidden_size, init_H, init_W)"""

    # H out​ =(H in​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    return nn.Sequential(
        nn.ConvTranspose2d(hidden_size, 64, 5, 2, padding=2, output_padding=1), # 8*2 + 3 -4+1 = 16
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(64, 32, 5, 2, padding=2, output_padding=1), # 16*2 + 3 -4 + 1 = 32
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(32, 32, 3, 2, padding=1, output_padding=1), # 32*2 +1 -2 +1 = 64
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(32, 2, 3, 1, padding=1), # 64 + 2 - 2 = 64
    )
    
def make_synmot_decoder(hidden_size):
    """make decoder
    decoder same as slot decoder. 
        using transposedConv2d 
        not using interpolate+conv2d
    
    `input`: size (BS*num_blocks, C=hidden_size, init_H, init_W)"""

    # H out​ =(H in​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    return nn.Sequential(
        nn.ConvTranspose2d(hidden_size, 64, 5, 2, padding=2, output_padding=1), # 8*2 + 3 -4+1 = 16
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(64, 64, 5, 2, padding=2, output_padding=1), # 16*2 + 3 -4 + 1 = 32
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1), # 32*2 +1 -2 +1 = 64
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(32, 4, 3, 1, padding=1), # 64 + 2 - 2 = 64
    )
    
def make_sprites_decoder(hidden_size):
    """
    decoder takes: [N, num_hidden, hidden_size] 
    outputs: [N, num_hidden, 3+1, 128, 128]
    """
    return nn.Sequential(
        nn.ConvTranspose2d(hidden_size, 128, 5, 2, padding=2, output_padding=1), # 8*2 + 3 -4+1 = 16
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1), # 16*2 + 3 -4 + 1 = 32
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(64, 32, 5, 2, padding=2, output_padding=1), # 32*2 + 3 -4 + 1 = 64
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(32, 32, 3, 2, padding=1, output_padding=1), # 64*2 +1 -2 +1 = 128
        nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(32, 4, 3, 1, padding=1), # 128 + 2 - 2 = 64
    )

def make_decoder_interp(hidden_size):
    """Method to initialize the decoder"""
    return nn.Sequential(
        LayerNorm(),
        nn.Conv2d(hidden_size, 64, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
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
        nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=0),
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
    # img = slots.repeat((1, 1, *resolution))
    img = slots.expand((-1, -1, *resolution))

    return img

class WrappedDecoder(nn.Module):
    """Decoder that takes slots embeddings and perform decoding individually and combine. 
    
    Args:
        `hidden_size`: the hidden size of the encoder
        `decoder`: the decoder type, either 'interp' or 'transconv'

    Input:
        `hidden`: (BS, K, d_slot) """
    def __init__(self, hidden_size, decoder='interp', mem_efficient=False, cell_switch=(), norm_method='default'):
        super().__init__()
        if decoder == 'synmot':
            self.decoder = make_synmot_decoder(hidden_size)
            self.out_channels = 3
        elif 'interp' in decoder:
            self.decoder = make_decoder_interp(hidden_size=hidden_size)
            self.out_channels = 1
        else:
            self.decoder = make_decoder_transconv(hidden_size=hidden_size)
            self.out_channels = 1
        self.pos_embed = SoftPositionEmbed(hidden_size, (8,8))
        self.hidden_size = hidden_size
        self.mem_efficient = mem_efficient
        self.cell_switch = cell_switch
        self.norm_method = 'default' if norm_method is None else norm_method
        if self.norm_method == 'normrelu':
            self.norm = NormReLU(dim=1)
        else:
            self.norm = torch.nn.Softmax(dim=1)

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        num_slots = hidden.shape[1]
        hidden = spatial_broadcast(hidden, (8,8)) # (BS*K, d_slot, 8, 8)
        hidden = self.pos_embed(hidden) # (BS*K, d_slot, 8, 8)
        if self.mem_efficient:
            dec_out_list = [
                self.decoder(broadcast_hidden_unit) \
                    for broadcast_hidden_unit in torch.chunk(hidden, num_slots, dim=0)
            ]
            dec_out = torch.cat(dec_out_list, dim=0) # Shape: [BS*self.hidden, 2, 64, 64]
        else:
            dec_out = self.decoder(hidden) # (BS*K, 2, 64, 64)
        channels, alpha_mask = unstack_and_split(dec_out, batch_size=batch_size, num_channels=self.out_channels) # (BS, K, *, H, W)
        channels = nn.Sigmoid()(channels)
        for cell_idx in self.cell_switch:
            alpha_mask[:, cell_idx, :, :] = float('-inf')
        alpha_mask = self.norm(alpha_mask) # (BS, <K>, 1, H, W)
        masked_channels = channels*alpha_mask
        fused = torch.sum(masked_channels, dim=1)

        return fused, channels, alpha_mask

class BroadcastConvDecoder(nn.Module):
    """
    Inputs:
        `z`: [N, latent_dim]
    Outputs:
        `slot`: [N, 3, image_size, image_size] normalized (sigmoid)
        `mask`: [N, 1, image_size, image_size] unnormalized (sigmoid)
        """
    def __init__(self, latent_dim, image_size=64):
        super().__init__()
        self.im_size = image_size + 8
        self.latent_dim = latent_dim
        self.init_grid()

        in_place = False
        self.g = nn.Sequential(
                    nn.Conv2d(self.latent_dim+2, 32, 3, 1, 0),
                    nn.ReLU(in_place),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(in_place),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(in_place),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(in_place),
                    nn.Conv2d(32, 4, 1, 1, 0),
                    nn.Sigmoid()
                    )

    def init_grid(self):
        x = torch.linspace(-1, 1, self.im_size)
        y = torch.linspace(-1, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
        
        
    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = self.broadcast(z)
        x = self.g(z) # [N, 4, imsize, imsize]
        slot = x[:, :3]
        # slot = nn.Sigmoid()(slot)
        mask = x[:, 3:]
        return slot, mask

if __name__ == '__main__':
    decoder = WrappedDecoder(100)

    # broadcast_hidden = torch.randn(2*6, 100, 8, 8)
    # hidden = torch.arange(2*6*100).reshape(2, 6, 100)
    hidden = torch.randn(2, 6, 100)

    out, *_ = decoder(hidden)
    print(out.shape)

    num_params = 0
    for p in decoder.decoder.parameters():
        num_params+=1
    for p in decoder.pos_embed.parameters():
        num_params+=1

    print(f"num of params: {num_params}")
    

    pass