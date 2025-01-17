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
        #in_features and out_features are the same
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
   
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
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

