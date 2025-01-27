import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MulitHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MulitHeadAttention, self).__init__()

        assert d_model%num_heads == 0 

        self.d_model = d_model # Dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of EACH head's K, Q and V

        #Weights to transform inputs
        #nn.Linear(in_features, out_features) are the same
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
   
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        #Dimensions are: {batch size, seq_length, num_heads, d_k}   

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) 
        #We swap num_heads and d_k so we can do a single parallel computation

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        '''
        x dimensions is {batch size, seq_length, num_heads, d_k} 
        we swap seq_length and num_heads for parallel computation
        '''
        batch_size, seq_length, d_k = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contigous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_k(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.combine_heads(self.W_o(attn_output))

        return output
    
class PositionWiseFeedForward (nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
