import torch
import torch.nn as nn
from torch.nn import functional as F
import Attention
import Encoding


# Feed Forward block
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        torch.nn.init.xavier_uniform_(self.linear1.weight.T) # FC layer weights initialized with Glorot initializer
        torch.nn.init.zeros_(self.linear1.bias)
        self.linear2 = nn.Linear(hidden, d_model)
        torch.nn.init.xavier_uniform_(self.linear2.weight.T) # FC layer weights initialized with Glorot initializer
        torch.nn.init.zeros_(self.linear2.bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x # (batch_size, seq_len, d)


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden):
        super(EncoderLayer, self).__init__()
        self.attention = Attention.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = FeedForward(d_model, ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
    
    def forward(self, x, src_mask=None):
        # Compute self attention
        a_output, _ = self.attention(Q=x, K=x, V=x, mask=src_mask)

        # Add and norm
        a_output = self.dropout1(a_output)
        ff_in = self.norm1(x + a_output)

        # Feed forward network
        ff_output = self.ffn(ff_in)

        # Add and norm
        ff_output = self.dropout2(ff_output)
        output = self.norm2(ff_in + ff_output) # (batch_size, input_sequence_length, d)
        return output

# Encoder layer with relative positional encoding
class ExtendedEncoderLayer(EncoderLayer):
    def __init__(self, d_model, num_heads, ffn_hidden, attention):
        super(ExtendedEncoderLayer, self).__init__(d_model, num_heads, ffn_hidden)
        if attention == "rel-e" or attention == "rel2-e":
            self.attention = Attention.MultiHeadAttention_relE(d_model, num_heads)
        elif attention == "rel-b" or attention == "rel2-b":
            self.attention = Attention.MultiHeadAttention_relB(d_model, num_heads)
        elif attention == "rel-eb" or attention == "rel2-eb":
            self.attention = Attention.MultiHeadAttention_relEB(d_model, num_heads)
        else:
            print("Error, insert a correct value")
        
    def forward(self, x, relative_ids, src_mask=None):
        # Compute self attention
        a_output, _ = self.attention(Q=x, K=x, V=x, relative_ids=relative_ids, mask=src_mask)

        # Add and norm
        a_output = self.dropout1(a_output)
        ff_in = self.norm1(x + a_output)

        # Feed forward network
        ff_output = self.ffn(ff_in)

        # Add and norm
        ff_output = self.dropout2(ff_output)
        output = self.norm2(ff_in + ff_output)
        return output


# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden):
        super(DecoderLayer, self).__init__()
        self.self_attention = Attention.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.cross_attention = Attention.MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = FeedForward(d_model, ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
    
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # Compute self attention
        a_output, _ = self.self_attention(Q=x, K=x, V=x, mask=trg_mask) # look ahead mask

        # Add and norm
        a_output = self.dropout1(a_output)
        out1 = self.norm1(x + a_output)

        # Compute encoder-decoder cross attention
        a_output2, _ = self.cross_attention(Q=out1, K=enc_output, V=enc_output, mask=src_mask) # padding mask

        # Add and norm
        a_output2 = self.dropout2(a_output2)
        ff_in = self.norm2(out1 + a_output2)
        
        # Feed forward network
        ff_output = self.ffn(ff_in)

        # Add and norm
        ff_output = self.dropout3(ff_output)
        output = self.norm3(ff_in + ff_output) # (batch_size, trg_seq_len, d)
        return output

# Decoder layer with relative positional encoding
class ExtendedDecoderLayer(DecoderLayer):
    def __init__(self, d_model, num_heads, ffn_hidden, attention="rel-eb"):
        super(ExtendedDecoderLayer, self).__init__(d_model, num_heads, ffn_hidden)
        
        if attention == "rel-e":
            self.self_attention = Attention.MultiHeadAttention_relE(d_model, num_heads)
        elif attention == "rel-b":
            self.self_attention = Attention.MultiHeadAttention_relB(d_model, num_heads)
        elif attention == "rel-eb":
            self.self_attention = Attention.MultiHeadAttention_relEB(d_model, num_heads)
        elif attention == "rel2-e":
            self.self_attention = Attention.MultiHeadAttention_relE(d_model, num_heads)
            self.cross_attention = Attention.MultiHeadAttention_relE(d_model, num_heads)
        elif attention == "rel2-b":
            self.self_attention = Attention.MultiHeadAttention_relB(d_model, num_heads)
            self.cross_attention = Attention.MultiHeadAttention_relB(d_model, num_heads)
        elif attention == "rel2-eb":
            self.self_attention = Attention.MultiHeadAttention_relEB(d_model, num_heads)
            self.cross_attention = Attention.MultiHeadAttention_relEB(d_model, num_heads)
        else:
            print("Error, insert a correct value")
        
        self.attention = attention

    
    def forward(self, x, enc_output, dec_relative_ids1, dec2enc_relative_ids, src_mask=None, trg_mask=None):
        # Compute self attention
        a_output, _ = self.self_attention(Q=x, K=x, V=x, relative_ids=dec_relative_ids1, mask=trg_mask)

        # Add and norm
        a_output = self.dropout1(a_output)
        out1 = self.norm1(x + a_output)

        # Compute encoder-decoder cross attention
        if self.attention == "rel-e" or self.attention == "rel-b" or self.attention == "rel-eb":
            a_output2, _ = self.cross_attention(Q=out1, K=enc_output, V=enc_output, mask=src_mask)
        else:
            a_output2, _ = self.cross_attention(Q=out1, K=enc_output, V=enc_output, relative_ids=dec2enc_relative_ids, mask=src_mask)

        # Add and norm
        a_output2 = self.dropout2(a_output)
        ff_in = self.norm2(out1 + a_output2)
        
        # Feed forward network
        ff_output = self.ffn(ff_in)

        # Add and norm
        ff_output = self.dropout3(ff_output)
        output = self.norm3(ff_in + ff_output)
        return output


# Encoder block
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_hidden) for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x 

# Encoder block with relative positional encoding   
class ExtendedEncoder(Encoder):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden, attention="rel-eb", shared_weights=False):
        super(ExtendedEncoder, self).__init__(n_layers, d_model, num_heads, ffn_hidden)
        if not shared_weights:
            self.layers = nn.ModuleList([ExtendedEncoderLayer(d_model, num_heads, ffn_hidden, attention) for _ in range(n_layers)])
        else:
            layer = ExtendedEncoderLayer(d_model, num_heads, ffn_hidden, attention)
            self.layers = nn.ModuleList([layer for _ in range(n_layers)])

    def forward(self, x, relative_ids, src_mask=None):
        for layer in self.layers:
            x = layer(x, relative_ids, src_mask)
        
        return x


# Decoder block
class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, ffn_hidden) for _ in range(n_layers)])
    
    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg = layer(trg, enc_src, src_mask, trg_mask)
        
        return trg

# Decoder block with relative positional encoding
class ExtendedDecoder(Decoder):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden, attention="rel-eb", shared_weights=False):
        super(ExtendedDecoder, self).__init__(n_layers, d_model, num_heads, ffn_hidden)
        if not shared_weights:
            self.layers = nn.ModuleList([ExtendedDecoderLayer(d_model, num_heads, ffn_hidden, attention) for _ in range(n_layers)])
        else:
            layer = ExtendedDecoderLayer(d_model, num_heads, ffn_hidden, attention)
            self.layers = nn.ModuleList([layer for _ in range(n_layers)])

    def forward(self, trg, enc_src, dec_relative_ids1, dec2enc_relative_ids, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg = layer(trg, enc_src, dec_relative_ids1, dec2enc_relative_ids, src_mask, trg_mask)
        
        return trg


# l: number of encoder/decoder layers
# d: dimensionality of token embeddings
# f: intermediate dimensionality used by the feed-forward sublayer
# h: number of attention-heads in the attention sublayers
# Standard Transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d, padding_idx=0)
        torch.nn.init.uniform_(self.encoder_embedding.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d, padding_idx=0)
        torch.nn.init.uniform_(self.decoder_embedding.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
        self.positional_encoding_enc = Encoding.AbsolutePositionalEncoding(d, max_seq_length_enc)
        self.positional_encoding_dec = Encoding.AbsolutePositionalEncoding(d, max_seq_length_dec)

        self.max_seq_length_enc = max_seq_length_enc
        self.max_seq_length_dec = max_seq_length_dec
        self.d = d

        self.encoder = Encoder(l, d, h, f)
        self.decoder = Decoder(l, d, h, f)

        self.fc = nn.Linear(d, trg_vocab_size)
        torch.nn.init.xavier_uniform_(self.fc.weight.T) # FC layer initialized with Glorot initializer
        torch.nn.init.zeros_(self.fc.bias)

        self.dropout_src = nn.Dropout(p=0.1)
        self.dropout_trg = nn.Dropout(p=0.1)

    def create_padding_mask(self, seq, device='cuda'):
        # Check where the sequence equals 0 (padding tokens), and cast to float
        mask = (seq == 0).float()  # (batch_size, seq_len)

        # Add extra dimensions to match the shape needed for attention mechanisms
        return mask[:, None, None, :].to(device)  # (batch_size, 1, 1, seq_len)
    
    def create_look_ahead_mask(self, size, device='cuda'):
        # Create a matrix with ones above the diagonal and zeros on and below the diagonal
        mask = torch.triu(torch.ones((size, size)), diagonal=1).float()  # (seq_len, seq_len)
        return mask.to(device) 

    def forward(self, src, trg):
        # source and target embedding and absolute positional encoding
        src_embedded = self.encoder_embedding(src) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        src_embedded = self.positional_encoding_enc(src_embedded)
        src_embedded = self.dropout_src(src_embedded)

        trg_embedded = self.decoder_embedding(trg) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        trg_embedded = self.positional_encoding_dec(trg_embedded)
        trg_embedded = self.dropout_trg(trg_embedded)
        # Encoder
        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask 
        enc_output = self.encoder(enc_output, src_mask)
        # Decoder
        dec_output = trg_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.shape[1])
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.maximum(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder.
                                                                        # It is used to pad and mask future tokens in the input received by the decoder.

        dec_output = self.decoder(dec_output, enc_output, trg_mask, src_mask)
        # output logits
        output = self.fc(dec_output)
        return output


# Transformer model with relative positional encoding
class ExtendedTransformer1(Transformer):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, attention="rel-eb"):
        super(ExtendedTransformer1, self).__init__(src_vocab_size, trg_vocab_size, d, h, l ,f, max_seq_length_enc, max_seq_length_dec)
        self.encoder = ExtendedEncoder(l, d, h, f, attention)
        self.decoder = ExtendedDecoder(l, d, h, f, attention)

        self.positional_encoding_enc = None
        self.positional_encoding_dec = None
    
    def forward(self, src, trg, enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids):
        # source and target embedding
        src_embedded = self.encoder_embedding(src) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        src_embedded = self.dropout_src(src_embedded)

        trg_embedded = self.decoder_embedding(trg) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        trg_embedded = self.dropout_trg(trg_embedded)
        # Encoder
        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask
        enc_output = self.encoder(enc_output, enc_relative_ids, src_mask)
        # Decoder
        dec_output = trg_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.shape[1])
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.maximum(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder.
                                                                        # It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, dec_relative_ids1, dec2enc_relative_ids, trg_mask, src_mask)
        # output logits
        output = self.fc(dec_output)
        return output


# Copy decoder block
class CopyDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CopyDecoder, self).__init__()
        self.attention = Attention.MultiHeadAttention(d_model, num_heads) # Scaled Dot Product Attention

        self.fcQ = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform_(self.fcQ.weight.T) # FC layer initialized with Glorot initializer
        torch.nn.init.zeros_(self.fcQ.bias)
        self.fcw = nn.Linear(d_model, 1)
        torch.nn.init.xavier_uniform_(self.fcw.weight.T) # FC layer initialized with Glorot initializer
        torch.nn.init.zeros_(self.fcw.bias)
    
    # p1 is the Transformer output without the Copy Decoder
    def forward(self, src_vocab_size, dec_output, enc_output, src, p1):
        # one hot encoding of source tokens -> value
        src_one_hot = F.one_hot(src, src_vocab_size)
        src_one_hot = src_one_hot.float()
        # learned query based on decoder outputs
        copy_query = self.fcQ(dec_output)
        # embeddings of last encoder layer are used as keys in the following attention
        a_output, _ = self.attention(Q=copy_query, K=enc_output, V=src_one_hot, copy=True) # no mask
        p2 = torch.softmax(a_output, dim=-1)
        # learnable weight 
        w = torch.sigmoid(self.fcw(dec_output))
        # output probability distribution
        output = (p2 * w) + ((1 - w) * p1)

        return output

# Transformer model with copy decoder and absolute positional encoding
class ExtendedStdTransformer2(Transformer):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec):
        super(ExtendedStdTransformer2, self).__init__(src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec)
        self.copy_decoder = CopyDecoder(d, h)

    def forward(self, src, trg, src_vocab_size):
        # source and target embedding and absolute positional encoding
        src_embedded = self.encoder_embedding(src) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        src_embedded = self.positional_encoding_enc(src_embedded)
        src_embedded = self.dropout_src(src_embedded)

        trg_embedded = self.decoder_embedding(trg) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        trg_embedded = self.positional_encoding_dec(trg_embedded)
        trg_embedded = self.dropout_trg(trg_embedded)
        # Encoder
        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask
        enc_output = self.encoder(enc_output, src_mask)
        # Decoder
        dec_output = trg_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.size(1))
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.maximum(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder. 
                                                                        # It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, trg_mask, src_mask)
        output = F.softmax(self.fc(dec_output), dim=-1)
        # Copy Decoder -> output probability distribution
        copy_output = self.copy_decoder(src_vocab_size, dec_output, enc_output, src, output)

        return copy_output

# Transformer model with copy decoder and relative positional encoding
class ExtendedTransformer2(ExtendedTransformer1):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, attention="rel-eb"):
        super(ExtendedTransformer2, self).__init__(src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, attention)
        self.copy_decoder = CopyDecoder(d, h)
    
    def forward(self, src, trg, enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids, src_vocab_size):
        # source and target embedding
        src_embedded = self.encoder_embedding(src) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        src_embedded = self.dropout_src(src_embedded)

        trg_embedded = self.decoder_embedding(trg) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        trg_embedded = self.dropout_trg(trg_embedded)
        # Encoder
        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask
        enc_output = self.encoder(enc_output, enc_relative_ids, src_mask)
        # Decoder
        dec_output = trg_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.size(1))
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.maximum(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder.
                                                                        # It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, dec_relative_ids1, dec2enc_relative_ids, trg_mask, src_mask)
        output = F.softmax(self.fc(dec_output), dim=-1)
        # Copy Decoder
        copy_output = self.copy_decoder(src_vocab_size, dec_output, enc_output, src, output)
        # output probability distribution
        return copy_output


# Transformer model with copy decoder, relative positional embedding and shared weights between encoder layers and decoder layers
class ExtendedTransformer4(ExtendedTransformer2):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, attention="rel-eb", shared_weights=True):
        super(ExtendedTransformer4, self).__init__(src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, attention)
        self.encoder = ExtendedEncoder(l, d, h, f, attention, shared_weights)
        self.decoder = ExtendedDecoder(l, d, h, f, attention, shared_weights)