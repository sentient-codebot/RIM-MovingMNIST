import torch
import torch.nn as nn

class GroupDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p<0 or p>1:
            raise ValueError("dropout probability illegal")
        self.p = torch.tensor(p).reshape(1)

    def forward(self, hidden: torch.Tensor, p = None) -> torch.Tensor:
        '''hidden:  (batch_size, num_units, dim_hidden)
            p:      tensor/list of (num_units,) or scalar or None
        '''
        if isinstance(p, list):
            p = torch.tensor(p)
        if self.training:
            if p is None:
                p = self.p
            
            if p.dim() != 1: # high-dim tensor: unaccepted 
                raise ValueError("illegal dimension of dropout probs")
            elif p.shape[0] > 1: # a vector: multiclass bernoulli
                assert p.shape[0] == hidden.shape[1]
                if False in [p_i>=0 and p_i<=1 for p_i in p]:
                    raise ValueError("dropout probability illegal")
                binomial = torch.distributions.binomial.Binomial(probs=1-p)
                compensate = 1./(1.-p)
            else: # a scalar: single class bernoulli
                if (p<0 or p>1).item():
                    raise ValueError("dropout probability illegal")
                binomial = torch.distributions.binomial.Binomial(probs=1-p)
                compensate = torch.ones(p.shape[0])/(1.-p)
            compensate = compensate.reshape((1,-1,1))
            rnd_mask = binomial.sample().reshape((1,-1,1))
            return hidden*rnd_mask*compensate

        return hidden

def main():
    inputs = torch.ones(4,6,3)
    dropout = GroupDropout(p=0.3)
    # probs = torch.tensor([0.1, 0.1, 0.4, 0.5, 0.6, 0.99])
    probs = None
    outputs = dropout(inputs, probs)
    print(outputs)

if __name__ == "__main__":
    main()