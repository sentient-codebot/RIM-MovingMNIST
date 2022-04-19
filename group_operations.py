import torch
import torch.nn as nn
import math


class GroupLinearLayer(nn.Module):
    """
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

class SharedWorkspace(nn.Module):
    """An implementation of "Coordination Among Neural Modules Through a Shared Global Workspace"(https://arxiv.org/abs/2103.01197) 

    Memory update is implemented with a skip connection, which is different than original methodology. 

    Args:
        `write_key_size`: attention key size for the memory writing phase
        `read_key_size`: attention key size for the broadcast and hidden state updating phase
        `memory_size`: memory slot size
        `hidden_size`: hidden state vector size
        `write_num_heads`: number of attention heads for write phase
        `read_num_heads`: number of attention heads for read phase
        `write_value_size`: value size for the memory writing phase
        `read_value_size`: value size for the broadcast and hidden state updating phase
        `write_dropout`: dropout probability for the memory writing phase
        `read_dropout`: dropout probability for the broadcast and hidden state updating phase

    Inputs:
        `M`: current memory slots of shape [N, K_mem, D_mem]
        `h`: hidden state vectors of shape [N, K_hidden, D_hidden] 
        `mask`: [optional] activation mask, select which hidden state vectors has write access to the memory slots, of shape [N, K_hidden]
        
    Output:
        `M_new`: updated memory slots
        `h`: updated hidden state vectors by broadcast memory slots """
    def __init__(self, write_key_size, read_key_size, memory_size, hidden_size, write_num_heads=1, read_num_heads=1, write_value_size=None, read_value_size=None, write_dropout=0.0, read_dropout=0.0):
        super().__init__()
        self.write_key_size = write_key_size
        self.read_key_size = read_key_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.write_num_heads = write_num_heads
        self.read_num_heads = read_num_heads
        self.write_value_size = write_value_size
        self.read_value_size = read_value_size
        self.write_dropout = write_dropout
        self.read_dropout = read_dropout
        if write_num_heads>1:
            assert write_value_size is not None 
        else:
            self.write_value_size = memory_size
        if read_num_heads>1:
            assert read_value_size is not None
        else: 
            self.read_value_size = hidden_size
        self.write_key_transform = nn.Linear(self.hidden_size, self.write_num_heads * self.write_key_size, bias=False) # hidden -> write_key
        self.write_value_transform = nn.Linear(self.hidden_size, self.write_num_heads * self.write_value_size, bias=False) # hidden -> write_value
        self.write_query_transform = nn.Linear(self.memory_size, self.write_num_heads * self.write_key_size, bias=False) # memory -> write_query
        self.write_output_transform = nn.Linear(self.write_num_heads * self.write_value_size, self.memory_size, bias=False) # write_value -> write_output
        self.write_dropout = nn.Dropout(self.write_dropout)

        self.read_key_transform = nn.Linear(self.memory_size, self.read_num_heads * self.read_key_size, bias=False) # memory -> read_key
        self.read_value_transform = nn.Linear(self.memory_size, self.read_num_heads * self.read_value_size, bias=False) # memory -> read_value
        self.read_query_transform = nn.Linear(self.hidden_size, self.read_num_heads * self.read_key_size, bias=False) # hidden -> read_query
        self.read_output_transform = nn.Linear(self.read_num_heads * self.read_value_size, self.hidden_size, bias=False) # read_value -> read_output
        self.read_dropout = nn.Dropout(self.read_dropout)

    def forward(self, M, h, mask=None):
        """
        Inputs:
            `M`: current memory slots of shape [N, K_mem, D_mem]
            `h`: hidden state vectors of shape [N, K_hidden, D_hidden] 
            `mask`: [optional] activation mask, select which hidden state vectors has write access to the memory slots, of shape [N, K_hidden]
            
        Output:
            `M_new`: updated memory slots
            `h_new`: updated hidden state vectors by broadcast memory slots """

        # Memory Writing Phase
        if mask is not None:
            mask = mask.squeeze() # Shape [N, num_hidden]
            assert mask.dim() == 2 
            mask = mask.view(mask.shape[0], 1, 1, mask.shape[1]) # Shape [N, 1, 1, num_hidden]
        write_key = self.write_key_transform(h) # Shape [N, K_hidden, write_num_heads * write_key_size]
        write_value = self.write_value_transform(h) # Shape [N, K_hidden, write_num_heads * D_mem]
        write_query = self.write_query_transform(M) # Shape [N, K_mem, write_num_heads * write_key_size]
        write_key = write_key.view(write_key.shape[0], self.write_num_heads, write_key.shape[1], self.write_key_size) # Shape [N, heads, K_hidden, write_key_size]
        write_value = write_value.view(write_value.shape[0], self.write_num_heads, write_value.shape[1], self.write_value_size) # Shape [N, heads, K_hidden, write_value_size]
        write_query = write_query.view(write_query.shape[0], self.write_num_heads, write_query.shape[1], self.write_key_size) # Shape [N, heads, K_mem, write_key_size]
        score_write = torch.matmul(write_query, write_key.transpose(-1,-2)) / math.sqrt(self.write_key_size) # Shape [N, heads, K_mem, K_hidden]
        score_write = score_write * mask # Shape [N, heads, K_mem, K_hidden]
        score_write = nn.Softmax(dim=3)(score_write) # Shape [N, heads, K_mem, K_hidden] 
        mem_update = torch.matmul(score_write, write_value) # Shape [N, heads, K_mem, write_value_size]
        mem_update = mem_update.transpose(1, 2).flatten(start_dim=-2) # Shape [N, K_mem, heads * write_value_size]
        mem_update = self.write_output_transform(self.write_dropout(mem_update)) # Shape [N, K_mem, D_mem]

        M_new = M + mem_update # Shape [N, K_mem, D_mem]

        # Memory Broadcast Phase and Hidden State Update Phase (Read Phase)
        read_key = self.read_key_transform(M_new) # Shape [N, K_mem, read_key_size]
        read_value = self.read_value_transform(M_new) # Shape [N, K_mem, D_hidden]
        read_query = self.read_query_transform(h) # Shape [N, K_hidden, read_key_size]
        read_key = read_key.view(read_key.shape[0], self.read_num_heads, read_key.shape[1], self.read_key_size) # Shape [N, heads, K_mem, read_key_size]
        read_value = read_value.view(read_value.shape[0], self.read_num_heads, read_value.shape[1], self.read_value_size) # Shape [N, heads, K_mem, read_value_size]
        read_query = read_query.view(read_query.shape[0], self.read_num_heads, read_query.shape[1], self.read_key_size) # Shape [N, heads, K_hidden, read_key_size]
        score_read = torch.matmul(read_query, read_key.transpose(-1,-2)) / math.sqrt(self.read_key_size) # Shape [N, heads, K_hidden, K_mem]
        score_read = nn.Softmax(dim=3)(score_read) # Shape [N, heads, K_hidden, K_mem]
        h_update = torch.matmul(score_read, read_value) # Shape [N, heads, K_hidden, read_value_size]
        h_update = h_update.transpose(1, 2).flatten(start_dim=-2) # Shape [N, K_hidden, heads * read_value_size]
        h_update = self.read_output_transform(self.read_dropout(h_update)) # Shape [N, K_hidden, D_hidden]

        h_new = h + h_update # Shape [N, K_hidden, D_hidden]

        return M_new, h_new


        