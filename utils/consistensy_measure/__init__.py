import torch

from .acl import average_consistent_length, maximum_consistent_length
from .img_corr import normalized_corr

def consistency_measure(
    input_seq: torch.Tensor,
    target_seq: torch.Tensor,
    corr_padding: tuple=(0, 0),
    output_ids: bool=True,
):
    """
    input:
        `input_seq`: [N, K1, T, C, H, W]
        `target_seq`: [N, K2, T, C, H, W]
        `corr_padding`: (h-wise, w-wise) padding for correlation operation
    """
    input_seq = input_seq.permute(2, 0, 1, 3, 4, 5) # [T, N, K1, C, H, W]
    target_seq = target_seq.permute(2, 0, 1, 3, 4, 5) # [T, N, K2, C, H, W]
    IDs = []
    for t in range(input_seq.shape[0]):
        corr_coef = normalized_corr(input_seq[t], target_seq[t], padding=corr_padding) # [N, K1, K2]
        _, indices = torch.max(corr_coef, dim=-1) # indices, shape [N, K1,]. 
        IDs.append(indices)
    IDs = torch.stack(IDs, dim=-1) # shape [N, 3, T]
    avr_len = average_consistent_length(IDs)
    max_len = maximum_consistent_length(IDs)
    
    if output_ids:
        return avr_len, max_len, IDs
    else:
        return avr_len, max_len