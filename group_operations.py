import torch
import torch.nn as nn
import math


class GroupLinearLayer(nn.Module):
    r"""
    for num_blocks blocks, do linear transformations independently

    self.w: (num_blocks, din, dout)

    x: (batch_size, num_blocks, din)
        -> permute: (num_blocks, batch_size, din)
        -> bmm with self.w: (num_blocks, batch_size, din) (bmm) (num_blocks, din, dout)
                            for each block in range(num_blocks):
                                do (batch_size, din) mat_mul (din, dout)
                                concatenate
                            result (num_blocks, batch_size, dout)
        -> permute: (batch_size, num_blocks, dout)

    """
    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """
    def __init__(self, inp_size, hidden_size, num_lstms):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        
        self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
        self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hid_state):
        """
        input: x (batch_size, num_lstms, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, num_lstms, hidden_state)
                c ((batch_size, num_lstms, hidden_state))
        """
        h, c = hid_state
        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :,  :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :,  3 * self.hidden_size:].tanh()
        i_t = gates[:, :,  :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t) 
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())#.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
        input: x (batch_size, num_grus, input_size)
               hidden (batch_size, num_grus, hidden_size)
        output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy


class GroupTorchGRU(nn.Module):
    '''
    Calculate num_units GRU cells in parallel
    '''
    def __init__(self, input_size, hidden_size, num_units):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_units = num_units
        gru_list = [nn.GRU(input_size=self.input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False) for _ in range(num_units)]
        self.grus = nn.ModuleList(gru_list)

    def forward(self, inputs, hidden):
        """
        input: x (batch_size, num_units, input_size)
               hidden (batch_size, num_units, hidden_size)
        output: hidden (batch_size, num_units, hidden_size)
        """

        hidden_list = [gru(inputs[:,i,:].unsqueeze(1), hidden[:,i,:].unsqueeze(0).contiguous())[1].squeeze(0) for i, gru in enumerate(self.grus)]
        # hidden_list: list of (batch_size, hidden_size)
        hidden_new = torch.stack(hidden_list, dim=1)

        return hidden_new


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
            compensate = compensate.reshape((1,-1,1)).to(hidden.device)
            rnd_mask = binomial.sample().reshape((1,-1,1)).to(hidden.device)
            return hidden*rnd_mask*compensate

        return hidden