import torch

def average_consistent_length(x):
    """
    input:
        `x`: a long tensor of shape [N, K, L]. batch size N, K trackers in parallel, each tracker has L frames, each frame has value 0 to K' as the object ID
    return:
        `avr_lengths`: a long tensor of shape [N, K]    
    """
    lengths, weights = _to_continuous_seq_length(x)
    average_length = torch.sum(weights * lengths, dim=-1) / torch.sum(weights, dim=-1) # Shape [N]
    return average_length

def maximum_consistent_length(x):
    """
    input:
        `x`: a long tensor of shape [N, K, L]"""
    lengths, weights = _to_continuous_seq_length(x) # Shape [N, L]
    maximum_length, _ = torch.max(lengths, dim=-1) # Shape [N]
    return maximum_length

def _to_continuous_seq_length(x, ):
    """ Calculate continuous sequence lengths inside a normal object-id sequence
    input:
        `x`: a long tensor of shape [N, K, L]
    output:
        `lengths`: a long tensor of shape [N, K, L']. L' is the maximum number of subsequences
        `weights`: a long tensor of shape [N, K, L'], signifying which elements in `lengths` are valid
        """
    y = torch.zeros_like(x)
    for t in range(1,x.shape[2]):
        flag = (x[:,:,t] != x[:,:,t-1]).to(torch.int) # [N, K], bool is_different
        y[:,:, t] = flag * (y[:,:, t-1]+1) + (1-flag) * y[:,:, t-1]
    lengths = []
    weights = []
    for seq_id in range(torch.max(y)):
        length = (y == seq_id).sum(dim=-1) # Shape [N, K]
        weight = (length > 1).to(torch.int) # For excluding 1 length sequence
        length = (length - 1.)/(x.shape[2]-1.) # Shape [N], normalize to 0-1, float
        lengths.append(length)
        weights.append(weight)
    lengths = torch.stack(lengths, dim=-1) # Shape [N, K, L']
    weights = torch.stack(weights, dim=-1) # Shape [N, K, L']
    
    return lengths, weights
    
        
        
def main():
    # dummy = torch.randint(0, 3, [1, 1, 10])
    dummy = torch.tensor([[0, 0, 0, 0, 0, 0, 4, 4, 4, 5,]]).unsqueeze(1)
    # dummy = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 4, 4,]]).unsqueeze(1)
    print(dummy)
    results = {}
    results['acl'] = average_consistent_length(dummy).squeeze(1)
    results['mcl'] = maximum_consistent_length(dummy).squeeze(1)
    
    for res in results.values():
        print(f'result: {res.item()*100: .2f}', '%')
    
if __name__ == "__main__":
    main()
        