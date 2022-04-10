import torch
import numpy as np
import torch.nn as nn

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)

class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer
        
        Args:
            `hidden_size`: Size of input feature dimension (channel)
            `resolution`: tuple of integers specifying height and width of grid (size)
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size, bias=True)
        self.grid = torch.tensor(data=build_grid(resolution)) # shape: [1, *resolution, 4]

    def forward(self, inputs):
        """soft positional embedding with learnable projection. 
        
        Input:
            `inputs`: size same as resolution [bs, C, *resolution] """
        pos_embed = self.dense(self.grid.to(inputs.device)) # shape: (1, *resolution, hidden_size)
        pos_embed = torch.movedim(pos_embed, -1, 1) # shape: (1, hidden_size, *resolution)
        return inputs + pos_embed # shape: (bs, hs, *resolution)

if __name__ == "__main__":
    pos_emb = SoftPositionEmbed(16, (4,8))

    image_embed = torch.rand((4, 16, 4, 8)) 
    img_pos_embed = pos_emb(image_embed)

    print('0')
