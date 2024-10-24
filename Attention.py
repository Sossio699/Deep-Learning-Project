# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import Encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Model dimension (d_model) has to be divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.dim_qkv = d_model // num_heads # Dimension of each head's key, query and value

        # Linear layers for transforming inputs
        self.w_q = nn.Linear(d_model, d_model) # Query
        #torch.nn.init.xavier_uniform_(self.w_q.weight) # Dense layer initialized with Glorot initializer
        self.w_k = nn.Linear(d_model, d_model) # Key
        #torch.nn.init.xavier_uniform_(self.w_k.weight) # Dense layer initialized with Glorot initializer
        self.w_v = nn.Linear(d_model, d_model) # Value
        #torch.nn.init.xavier_uniform_(self.w_v.weight) # Dense layer initialized with Glorot initializer
        self.w_o = nn.Linear(d_model, d_model) # Output
        #torch.nn.init.xavier_uniform_(self.w_o.weight) # Dense layer initialized with Glorot initializer

        #self.scale = torch.sqrt(torch.FloatTensor([self.dim_qkv]))
    
    def scaled_dot_product_attention(self, Q, K, V, mask, device='cuda'):
        # Compute attention scores
        #print(Q.shape)
        #print(K.shape)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        dk = K.size(-1)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        #print(attention_scores.shape)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            #print(attention_scores.shape)
            #print(f"mask shape: {mask.shape}")
            #attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores += (mask * 1e-9)
        
        # Softmax is applied to obtain attention probabilities
        attention_probabilities = F.softmax(attention_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attention_probabilities, V)
        return output, attention_scores

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, _, _ = x.size()
        return x.view(batch_size, -1, self.num_heads, self.dim_qkv).permute(0, 2, 1, 3)
    
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, _, _ = x.size()
        return x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, copy=False, mask=None):
        # If copy is True performs only Scaled dot-product Attention
        if not copy:
            # Apply linear transformations and split heads
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
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention_relE, self).__init__(d_model, num_heads)
        self.max_relative_position = 16 # As indicated in the paper
        self.relative_vocab_size = self.max_relative_position * 2 + 1
        self.relative_embeddings = nn.Embedding(self.relative_vocab_size, self.dim_qkv)
        #torch.nn.init.uniform_(self.relative_embeddings.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
    
    def scaled_dot_product_attention(self, Q, K, V, relative_ids, mask, device='cuda'):
        #print(Q.shape)
        #print(K.shape)
        # Compute attention scores
        mat_qk = torch.matmul(Q, K.transpose(-2, -1))
        #print(mat_qk.shape)
        rp_k = self.relative_embeddings(relative_ids.to(device))
        #print(rp_k.shape)
        mat_qr = torch.einsum("bhqd,qkd->bhqk", Q, rp_k) # Einstein summation
        #print(mat_qr.shape)
        attention_scores = (mat_qk + mat_qr)  # Attention scores with relative positional encoding
        #print(f"Attention scores: {attention_scores.shape}")

        dk = K.size(-1)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores += (mask * 1e-9)
        
        # Softmax is applied to obtain attention probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Multiply by values to obtain the final output
        #print(f"Attention: {attention_weights.shape}")
        #print(f"V: {V.shape}")
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
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention_relB, self).__init__(d_model, num_heads)
        self.max_relative_position = 16 # As indicated in the paper
        self.relative_vocab_size = self.max_relative_position * 2 + 1
        self.relative_bias = nn.Embedding(self.relative_vocab_size, 1)
        #torch.nn.init.uniform_(self.relative_bias.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
    
    def scaled_dot_product_attention(self, Q, K, V, relative_ids, mask, device='cuda'):
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        bias = self.relative_bias(relative_ids.to(device))
        attention_scores += torch.squeeze(bias, -1)

        dk = K.size(-1)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores += (mask * 1e-9)
        
        # Softmax is applied to obtain attention probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

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
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention_relEB, self).__init__(d_model, num_heads)
        self.max_relative_position = 16 # As indicated in the paper
        self.relative_vocab_size = self.max_relative_position * 2 + 1
        self.relative_embeddings = nn.Embedding(self.relative_vocab_size, self.dim_qkv)
        #torch.nn.init.uniform_(self.relative_embeddings.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
        self.relative_bias = nn.Embedding(self.relative_vocab_size, 1)
        #torch.nn.init.uniform_(self.relative_bias.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
    
    def scaled_dot_product_attention(self, Q, K, V, relative_ids, mask, device='cuda'):
        # Compute attention scores
        #print(Q.shape)
        #print(K.shape)
        mat_qk = torch.matmul(Q, K.transpose(-2, -1))
        rp_k = self.relative_embeddings(relative_ids.to(device))
        #print(rp_k.shape)
        #print(Q.shape)
        mat_qr = torch.einsum("bhqd,qkd->bhqk", Q, rp_k) # Einstein summation
        attention_scores = (mat_qk + mat_qr) # Attention scores with relative positional encoding

        bias = self.relative_bias(relative_ids.to(device))
        attention_scores = attention_scores + torch.squeeze(bias, -1)

        dk = K.shape[-1]
        attention_scores = attention_scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attention_scores += (mask * 1e-9)
        
        # Softmax is applied to obtain attention probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

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