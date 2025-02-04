import torch
import torch.nn as nn
import math
import numpy as np

# Sine and cosine functions with different frequencies to generate the positional encoding
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(AbsolutePositionalEncoding, self).__init__()

        positional_encodings = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', positional_encodings.unsqueeze(0))
        # Registered as a buffer -> part of the module's state, but not a trainable parameter
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# If cross attention ids are needed (rel2-*), then dec2enc_ids = True
# Function to compute relative positional encodings
def create_relative_ids(inp_len, trg_len, max_relative_position, dec2enc_ids):
    enc_relative_ids = np.zeros([inp_len, inp_len], dtype=int)
    for i in range(inp_len):
        for j in range(inp_len):
            diff = i - j
            diff = max_relative_position + min(max(diff, -max_relative_position), max_relative_position)
            enc_relative_ids[i][j] = diff
    
    dec_relative_ids1 = np.zeros([trg_len - 1, trg_len - 1], dtype=int)
    for i in range(trg_len - 1):
        for j in range(trg_len - 1):
            diff = i - j
            diff = max_relative_position + min(max(diff, -max_relative_position), max_relative_position)
            dec_relative_ids1[i][j] = diff
    
    dec2enc_relative_ids = np.zeros([trg_len - 1, inp_len], dtype=int)
    for i in range(trg_len - 1):
        for j in range(inp_len):
            if dec2enc_ids:
                diff = i - j
                diff = max_relative_position + min(max(diff, -max_relative_position), max_relative_position)
                dec2enc_relative_ids[i][j] = diff
            else:
                dec2enc_relative_ids[i][j] = max_relative_position
    
    return (torch.IntTensor(enc_relative_ids), torch.IntTensor(dec_relative_ids1), torch.IntTensor(dec2enc_relative_ids))