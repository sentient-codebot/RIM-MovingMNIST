import torch
import torch.nn as nn
import math
from functools import wraps
from typing import Optional
from torch import Tensor


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

class SharedGroupLinearLayer(nn.Module):
    """All the parameters are shared using soft attention this layer is used for sharing Q,K,V parameters of MHA
    
    Outputs are soft weighted sum of all layer blocks. 
    """
    kdim = 16
    def __init__(self, din, dout, n_templates):
        super(SharedGroupLinearLayer, self).__init__()

        self.w = nn.ParameterList([nn.Parameter(0.01 * torch.randn(din,dout)) for _ in range(0,n_templates)])
        self.gll_write = GroupLinearLayer(dout,self.kdim, n_templates) # 16 == key size
        self.gll_read = GroupLinearLayer(din,self.kdim,1)

    def forward(self,x):
        #input size (bs,num_blocks,din), required matching num_blocks vs n_templates
        bs_size = x.shape[0]
        k = x.shape[1]
        x= x.reshape(k*bs_size,-1)
        x_read = self.gll_read((x*1.0).reshape((x.shape[0], 1, x.shape[1]))) # for q, shape [k*bs, 1, 16]
        x_next = []
        for weight in self.w:
            x_next_l = torch.matmul(x,weight) # Shape: [batch_size*num_blocks, dout]
            x_next.append(x_next_l)
        x_next = torch.stack(x_next,1) #(k*bs,n_templates,dout)
        
        x_write = self.gll_write(x_next) # for k,v, shape: [k*bs, n_templates, 16]
        sm = nn.Softmax(2)
        att = sm(torch.bmm(x_read, x_write.permute(0, 2, 1))) # Shape: [k*bs, 1, *n_templates]
        
        x_next = torch.bmm(att, x_next)

        x_next = x_next.mean(dim=1).reshape(bs_size,k,-1)
        
        return x_next




class SharedBlockGRU(nn.Module):
    """Dynamic sharing of parameters between blocks(RIM's)
    
    Args:
        `ninp`: input dimension (total dimension of all blocks concatenated), should be dividable by `k`
        `nhid`: hidden dimension (total dimension of all blocks concatenated), should be dividable by `k`
        `k`: number of blocks=object files
        `n_templates`: number of templates"""

    def __init__(self, ninp, nhid, num_hidden, n_templates):
        super(SharedBlockGRU, self).__init__()

        assert ninp % num_hidden == 0 
        assert nhid % num_hidden == 0

        self.num_hidden = num_hidden
        self.single_hidden_size = nhid // self.num_hidden # dimension of each block's hidden state

        self.n_templates = n_templates
        self.templates = nn.ModuleList([nn.GRUCell(ninp//self.num_hidden,self.single_hidden_size) for _ in range(0,self.n_templates)]) # GRUCell: input_size: ninp, hidden_size: m
        self.nhid = nhid

        self.ninp = ninp

        self.gll_write = GroupLinearLayer(self.single_hidden_size,16, self.n_templates) # for k, 16 == key size
        self.gll_read = GroupLinearLayer(self.single_hidden_size,16,1) # for q, 16 == key size
        print("Using Gumble sparsity in", __class__.__name__)

    def blockify_params(self):

        return

    def forward(self, input, h):
        """
        Inputs:
            `input`: [N, num_hidden*single_input_size]
            `h`: [N, num_hidden*single_hidden_size]
            
        Outputs:
            `hnext`: [N, num_hidden*single_hidden_size],
            `attn`: [N, num_OFs, n_templates] (num_bloccks==k==num_object_files)
        """

        #self.blockify_params()
        bs = h.shape[0]                                                                      # h: previous hidden state  
        h = h.reshape((h.shape[0], self.num_hidden, self.single_hidden_size)).reshape((h.shape[0]*self.num_hidden, self.single_hidden_size))     # h_shape: (h.shape[0]*self.num_hidden, self.m)

        # following two lines are problematica. shouldn't copy input for num_hidden times, we just need to split it
        # input = input.reshape(input.shape[0], 1, input.shape[1])                            # Shape: [N, 1, num_hidden*din]
        # input = input.repeat(1,self.num_hidden,1)                                           # Shape: [N, num_hidden, num_hidden*din]
        input = input.reshape(input.shape[0], self.num_hidden, -1)                          # Shape: [N, num_hidden, din]
        input = input.reshape(input.shape[0]*self.num_hidden, -1)                           # Shape: [N*num_hidden, din]

        h_read = self.gll_read((h*1.0).reshape((h.shape[0], 1, h.shape[1])))                # from current hidden to q, shape: [N*num_hidden, 1, 16]


        hnext_stack = []


        for template in self.templates:                                                     # input [N*num_hidden, num_hidden*din], h [N*num_hidden, m]
            hnext_l = template(input, h)                                                    # Shape: [N*num_hidden, m]
            hnext_l = hnext_l.reshape((hnext_l.shape[0], 1, hnext_l.shape[1]))              # Shape: [N*num_hidden, 1, m]
            hnext_stack.append(hnext_l)

        hnext = torch.cat(hnext_stack, 1)                                                   # Shape: [N*num_hidden, n_templates, m]

        write_key = self.gll_write(hnext)                                                   # from candidate hidden to k, shape: [N*num_hidden, n_templates, 16]

        att = torch.nn.functional.gumbel_softmax(torch.bmm(h_read, write_key.permute(0, 2, 1)),  tau=0.5, hard=True)    # Shape: [N*num_hidden, 1, n_templates]

        #att = att*0.0 + 0.25

        #print('hnext shape before att', hnext.shape)
        hnext = torch.bmm(att, hnext)   #((hnext_l.shape[0], 1, hnext_l.shape[1]))
        hnext = hnext.mean(dim=1)
        hnext = hnext.reshape((bs, self.num_hidden, self.single_hidden_size)).reshape((bs, self.num_hidden*self.single_hidden_size))
        #print('shapes', hnext.shape, cnext.shape)

        return hnext, att.data.reshape(bs,self.num_hidden,self.n_templates)

class SharedGroupGRU(nn.Module):
    """Dynamic sharing of parameters (GRU) between hidden state vectors.
    
    Args:
        `input_size`: dimension of a single input
        `hidden_size`: dimension of a single hidden state vector
        `num_hidden`:
        `num_rules`:
        `use_rule_embedding`: bool = False
    Inputs:
        `input`: [N, num_hidden, single_input_size]
        `h`: [N, num_hidden, single_hidden_size]
        
    Outputs:
        `h_new`: [N, num_hidden, single_hidden_size],
        `attn`: [N, num_hidden, num_rules] 
    """
    key_size = 64
    def __init__(self, input_size: int, hidden_size: int, num_hidden: int, num_rules: int, use_rule_embedding: bool=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size # dimension of each block's hidden state

        self.num_hidden = num_hidden

        self.num_rules = num_rules
        self.rules = nn.ModuleList([nn.GRUCell(self.input_size,self.hidden_size) for _ in range(0,self.num_rules)]) # GRUCell: input_size: ninp, hidden_size: m

        self.use_rule_embedding = use_rule_embedding # we can use a higher rule_emb size and then project to kdim, but also directly use kdim
        if not self.use_rule_embedding:
            self.gll_read = GroupLinearLayer(self.hidden_size,self.key_size,1) # hidden -> q, 16 == key size
            # self.gll_write = GroupLinearLayer(self.hidden_size,self.key_size, self.num_rules) # hidden_new -> k
            self.gll_write = nn.Linear(self.hidden_size, self.key_size, bias=False) # hidden_new -> k, sharing same key proj matrix
        else:
            self.gll_read = GroupLinearLayer(self.input_size+self.hidden_size,self.key_size, 1) # input+hidden -> q, 16 == key size
            self.rule_embeddings = nn.Parameter(torch.randn(1, self.num_rules, self.key_size)) # Shape: [1, num_rules, key_size]
            print('Use rule embedding in', __class__.__name__) 

        print("Using Gumble sparsity in", __class__.__name__)

    def forward(self, input, h):
        """
        Inputs:
            `input`: [N, num_hidden, single_input_size]
            `h`: [N, num_hidden, single_hidden_size]
            
        Outputs:
            `hnext`: [N, num_hidden, single_hidden_size],
            `attn`: [N, num_OFs, n_templates] (num_bloccks==k==num_object_files)
        """

        #self.blockify_params()
        bs = h.shape[0]                                                                      # h: previous hidden state  
        h = h.reshape((h.shape[0]*self.num_hidden, self.hidden_size))   # h.shape: (bs*num_hidden, hidden_size)

        input = input.reshape(input.shape[0]*self.num_hidden, self.input_size)  # Shape: [N*num_hidden, input_size]

        if not self.use_rule_embedding:
            h_read = self.gll_read((h*1.0).reshape((h.shape[0], 1, h.shape[1]))) # from current hidden to q, shape: [N*num_hidden, 1, kdim]
        else:
            h_read = self.gll_read(torch.cat((input, h), dim=1).unsqueeze(1)) # from input+hidden to q, shape: [N*num_hidden, 1, kdim]


        hnext_stack = []


        for rule in self.rules:         # input [N*num_hidden, input_size], h [N*num_hidden, hidden_size]
            hnext_l = rule(input, h)    # Shape: [N*num_hidden, hidden_size]
            hnext_l = hnext_l.reshape((hnext_l.shape[0], 1, hnext_l.shape[1])) # Shape: [N*num_hidden, 1, hidden_size]
            hnext_stack.append(hnext_l)

        hnext = torch.cat(hnext_stack, 1) # Shape: [N*num_hidden, num_rules, hidden_sixe]

        if not self.use_rule_embedding:
            write_key = self.gll_write(hnext) # from candidate hidden to k, shape: [N*num_hidden, num_rules, key_size]
        else:
            write_key = self.rule_embeddings # [1, num_rules, kdim]

        att = torch.nn.functional.gumbel_softmax(torch.matmul(h_read, write_key.permute(0, 2, 1)),  tau=0.5, hard=True)    # Shape: [N*num_hidden, 1, num_rules]

        #print('hnext shape before att', hnext.shape)
        hnext = torch.bmm(att, hnext)   # [N*num_hidden, 1, num_rules], [N*num_hidden, num_rules, hidden_size] -> [N*num_hidden, 1, hidden_size]
        hnext = hnext.mean(dim=1) # [N*num_hidden, hidden_size]
        hnext = hnext.reshape((bs, self.num_hidden, self.hidden_size)) # [N, num_hidden, hidden_size]
        #print('shapes', hnext.shape, cnext.shape)

        return hnext, att.data.reshape(bs,self.num_hidden,self.num_rules)
    
class SharedBlockLSTM(nn.Module):
    """Dynamic sharing of parameters between blocks(RIM's)
    
    Inputs:
        `input`: [N, num_hidden*single_input_size]
        `h`: [N, num_hidden*single_hidden_size]
        `c`: [N, num_hidden*single_hidden_size]
    """

    def __init__(self, ninp, nhid, num_hidden , n_templates):
        super(SharedBlockLSTM, self).__init__()

        assert ninp % num_hidden == 0
        assert nhid % num_hidden == 0

        self.num_hidden = num_hidden
        self.single_hidden_size = nhid // self.num_hidden
        self.n_templates = n_templates
        self.templates = nn.ModuleList([nn.LSTMCell(ninp//self.num_hidden,self.single_hidden_size) for _ in range(0,self.n_templates)])
        self.nhid = nhid

        self.ninp = ninp

        self.gll_write = GroupLinearLayer(self.single_hidden_size,16, self.n_templates)
        self.gll_read = GroupLinearLayer(self.single_hidden_size,16,1)

    def blockify_params(self):

        return

    def forward(self, input, h, c):

        #self.blockify_params()
        bs = h.shape[0]
        h = h.reshape((h.shape[0], self.num_hidden, self.single_hidden_size)).reshape((h.shape[0]*self.num_hidden, self.single_hidden_size))
        c = c.reshape((c.shape[0], self.num_hidden, self.single_hidden_size)).reshape((c.shape[0]*self.num_hidden, self.single_hidden_size))


        input = input.reshape(input.shape[0], self.num_hidden, -1) # Shape: [N, num_hidden, din]
        input = input.reshape(input.shape[0]*self.num_hidden, input.shape[2]) # Shape: [N*num_hidden, din]

        h_read = self.gll_read((h*1.0).reshape((h.shape[0], 1, h.shape[1])))


        hnext_stack = []
        cnext_stack = []


        for template in self.templates:#[self.lstm1, self.lstm2, self.lstm3, self.lstm4]:
            hnext_l, cnext_l = template(input, (h, c))

            hnext_l = hnext_l.reshape((hnext_l.shape[0], 1, hnext_l.shape[1]))
            cnext_l = cnext_l.reshape((cnext_l.shape[0], 1, cnext_l.shape[1]))

            hnext_stack.append(hnext_l)
            cnext_stack.append(cnext_l)

        hnext = torch.cat(hnext_stack, 1)
        cnext = torch.cat(cnext_stack, 1)


        write_key = self.gll_write(hnext)

        sm = nn.Softmax(2)
        att = sm(torch.bmm(h_read, write_key.permute(0, 2, 1)))

        #att = att*0.0 + 0.25

        #print('hnext shape before att', hnext.shape)
        hnext = torch.bmm(att, hnext)
        cnext = torch.bmm(att, cnext)

        hnext = hnext.mean(dim=1)
        cnext = cnext.mean(dim=1)

        hnext = hnext.reshape((bs, self.num_hidden, self.single_hidden_size)).reshape((bs, self.num_hidden*self.single_hidden_size))
        cnext = cnext.reshape((bs, self.num_hidden, self.single_hidden_size)).reshape((bs, self.num_hidden*self.single_hidden_size))

        #print('shapes', hnext.shape, cnext.shape)

        return hnext, cnext, att.data.reshape(bs,self.num_hidden,self.n_templates)

class SharedGRUCell(nn.GRUCell):
    """passing multile (h, x) in paralell to one shared GRU
    """
    @wraps(nn.GRUCell.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, input: Tensor, hx: Optional[Tensor] = None):
        """
        input: [N, num_hidden, single_input_size]
        hx: [N, num_hidden, single_hidden_size]
        """
        h_new = super().forward(
            input.view(input.shape[0]*input.shape[1], input.shape[2]),
            hx.view(hx.shape[0]*hx.shape[1], hx.shape[2])
        )
        h_new = h_new.view(hx.shape[0], hx.shape[1], hx.shape[2])
        return h_new
    