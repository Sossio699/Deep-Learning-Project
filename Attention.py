# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import Encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        # Model dimension (d_model) has to be divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.dim_qkv = d_model // num_heads # Dimension of each head's key, query and value

        # Linear layers for transforming inputs
        self.w_q = nn.Linear(d_model, d_model) # Query
        self.w_k = nn.Linear(d_model, d_model) # Key
        self.w_v = nn.Linear(d_model, d_model) # Value
        self.w_o = nn.Linear(d_model, d_model) # Output

        self.scale = torch.sqrt(torch.FloatTensor([self.dim_qkv]))
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask):
        # Compute attention scores
        print(Q.shape)
        print(K.shape)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        print(attention_scores.shape)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attention_probabilities = self.dropout(torch.softmax(attention_scores, dim=-1))

        # Multiply by values to obtain the final output
        output = torch.matmul(attention_probabilities, V)
        return output, attention_scores

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.dim_qkv).transpose(1, 2)
    
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, dim_qkv = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, copy=False, mask=None):
        # Apply linear transformations and split heads
        if not copy:
            Q = self.split_heads(self.w_q(Q))
            K = self.split_heads(self.w_k(K))
            V = self.split_heads(self.w_v(V))

        #Perform scaled dot-product attention
        attention_output, attention_scores = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        if not copy:
            output = self.w_o(self.combine_heads(attention_output))
        else:
            output = attention_output
        return output, attention_scores


class MultiHeadAttention_relE(MultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention_relE, self).__init__(d_model, num_heads, dropout)
        self.max_relative_position = 16 # As indicated in the paper
        self.relative_vocab_size = self.max_relative_position * 2 + 1
        self.relative_embeddings = nn.Embedding(self.relative_vocab_size, d_model // num_heads)
    
    def scaled_dot_product_attention(self, Q, K, V, relative_ids, mask):
        # Compute attention scores
        mat_qk = torch.matmul(Q, K.transpose(-2, -1))
        rp_k = self.relative_embeddings(relative_ids)
        mat_qr = torch.einsum("bhqd,qkd->bhqk", Q, rp_k) # Einstein summation
        attention_scores = (mat_qk + mat_qr) / self.scale # Scaled attention scores with relative positional encoding

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attention_weights = self.dropout(torch.softmax(attention_scores, dim=-1))

        # Multiply by values to obtain the final output
        output = torch.matmul(attention_weights, V)
        return output, attention_scores

    def forward(self, Q, K, V, relative_ids, mask):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.w_q(Q))
        K = self.split_heads(self.w_k(K))
        V = self.split_heads(self.w_v(V))
        
        #Perform scaled dot-product attention
        attention_output, attention_scores = self.scaled_dot_product_attention(Q, K, V, relative_ids, mask)

        # Combine heads and apply output transformation
        output = self.w_o(self.combine_heads(attention_output))
        return output, attention_scores


class MultiHeadAttention_relB(MultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention_relB, self).__init__(d_model, num_heads, dropout)
        self.max_relative_position = 16 # As indicated in the paper
        self.relative_vocab_size = self.max_relative_position * 2 + 1
        self.relative_bias = nn.Embedding(self.relative_vocab_size, 1)
    
    def scaled_dot_product_attention(self, Q, K, V, relative_ids, mask):
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        bias = self.relative_bias(relative_ids)
        attention_scores = attention_scores + torch.squeeze(bias, -1)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attention_weights = self.dropout(torch.softmax(attention_scores, dim=-1))

        # Multiply by values to obtain the final output
        output = torch.matmul(attention_weights, V)
        return output, attention_scores

    def forward(self, Q, K, V, relative_ids, mask):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.w_q(Q))
        K = self.split_heads(self.w_k(K))
        V = self.split_heads(self.w_v(V))

        #Perform scaled dot-product attention
        attention_output, attention_scores = self.scaled_dot_product_attention(Q, K, V, relative_ids, mask)

        # Combine heads and apply output transformation
        output = self.w_o(self.combine_heads(attention_output))
        return output, attention_scores


class MultiHeadAttention_relEB(MultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention_relEB, self).__init__(d_model, num_heads, dropout)
        self.max_relative_position = 16 # As indicated in the paper
        self.relative_vocab_size = self.max_relative_position * 2 + 1
        self.relative_embeddings = nn.Embedding(self.relative_vocab_size, d_model // num_heads)
        self.relative_bias = nn.Embedding(self.relative_vocab_size, 1)
    
    def scaled_dot_product_attention(self, Q, K, V, relative_ids, mask):
        # Compute attention scores
        print(Q.shape)
        print(K.shape)
        mat_qk = torch.matmul(Q, K.transpose(-2, -1))
        rp_k = self.relative_embeddings(relative_ids)
        print(rp_k.shape)
        print(Q.shape)
        mat_qr = torch.einsum("bhqd,qkd->bhqk", Q, rp_k) # Einstein summation
        attention_scores = (mat_qk + mat_qr) / self.scale # Scaled attention scores with relative positional encoding

        bias = self.relative_bias(relative_ids)
        attention_scores = attention_scores + torch.squeeze(bias, -1)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attention_weights = self.dropout(torch.softmax(attention_scores, dim=-1))

        # Multiply by values to obtain the final output
        output = torch.matmul(attention_weights, V)
        return output, attention_scores
    
    def forward(self, Q, K, V, relative_ids, mask):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.w_q(Q))
        K = self.split_heads(self.w_k(K))
        V = self.split_heads(self.w_v(V))

        #Perform scaled dot-product attention
        attention_output, attention_scores = self.scaled_dot_product_attention(Q, K, V, relative_ids, mask)

        # Combine heads and apply output transformation
        output = self.w_o(self.combine_heads(attention_output))
        return output, attention_scores