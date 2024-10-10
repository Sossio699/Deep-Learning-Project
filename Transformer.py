# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import Attention
import Encoding

# Utilities for masks
def look_ahead_mask(size):
  mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

def padding_mask(seq):
    return 0


# FC layers are applied along the last (512) dimension
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        torch.nn.init.xavier_uniform_(self.linear1.weight) # Dense layer initialized with Glorot initializer
        self.linear2 = nn.Linear(hidden, d_model)
        torch.nn.init.xavier_uniform_(self.linear2.weight) # Dense layer initialized with Glorot initializer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = Attention.MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        # Compute self attention
        a_output, _ = self.attention(Q=x, K=x, V=x, mask=src_mask)

        # Add and norm
        a_output = self.dropout(a_output)
        x = self.norm1(x + a_output)

        # Feed forward network
        ff_output = self.ffn(x)

        # Add and norm
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)
        return x

class ExtendedEncoderLayer(EncoderLayer):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, attention):
        super(ExtendedEncoderLayer, self).__init__(d_model, num_heads, ffn_hidden, dropout)
        if attention == "rel-e" or attention == "rel2-e":
            self.attention = Attention.MultiHeadAttention_relE(d_model, num_heads, dropout)
        elif attention == "rel-b" or attention == "rel2-b":
            self.attention = Attention.MultiHeadAttention_relB(d_model, num_heads, dropout)
        elif attention == "rel-eb" or attention == "rel2-eb":
            self.attention = Attention.MultiHeadAttention_relEB(d_model, num_heads, dropout)
        else:
            print("Error, insert a correct value")
        
    def forward(self, x, relative_ids, src_mask=None):
        # Compute self attention
        a_output, _ = self.attention(Q=x, K=x, V=x, relative_ids=relative_ids, mask=src_mask)

        # Add and norm
        a_output = self.dropout(a_output)
        x = self.norm1(x + a_output)

        # Feed forward network
        ff_output = self.ffn(x)

        # Add and norm
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = Attention.MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = Attention.MultiHeadAttention(d_model, num_heads,dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # Compute self attention
        a_output, _ = self.self_attention(Q=x, K=x, V=x, mask=trg_mask)

        # Add and norm
        a_output = self.dropout(a_output)
        x = self.norm1(x + a_output)

        if enc_output is not None:
            # Compute encoder-decoder cross attention
            a_output, _ = self.cross_attention(Q=x, K=enc_output, V=enc_output, mask=src_mask)

            # Add and norm
            a_output = self.dropout(a_output)
            x = self.norm2(x + a_output)
        
        # Feed forward network
        ff_output = self.ffn(x)

        # Add and norm
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)
        return x

class ExtendedDecoderLayer(DecoderLayer):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, attention):
        super(ExtendedDecoderLayer, self).__init__(d_model, num_heads, ffn_hidden, dropout)
        if attention == "rel-e":
            self.self_attention = Attention.MultiHeadAttention_relE(d_model, num_heads, dropout)
        elif attention == "rel-b":
            self.self_attention = Attention.MultiHeadAttention_relB(d_model, num_heads, dropout)
        elif attention == "rel-eb":
            self.self_attention = Attention.MultiHeadAttention_relEB(d_model, num_heads, dropout)
        elif attention == "rel2-e":
            self.self_attention = Attention.MultiHeadAttention_relE(d_model, num_heads, dropout)
            self.cross_attention = Attention.MultiHeadAttention_relE(d_model, num_heads, dropout)
        elif attention == "rel2-b":
            self.self_attention = Attention.MultiHeadAttention_relB(d_model, num_heads, dropout)
            self.cross_attention = Attention.MultiHeadAttention_relB(d_model, num_heads, dropout)
        elif attention == "rel2-eb":
            self.self_attention = Attention.MultiHeadAttention_relEB(d_model, num_heads, dropout)
            self.cross_attention = Attention.MultiHeadAttention_relEB(d_model, num_heads, dropout)
        else:
            print("Error, insert a correct value")
        
        self.check = attention

    
    def forward(self, x, enc_output, dec_relative_ids1, dec2enc_relative_ids, src_mask=None, trg_mask=None):
        # Compute self attention
        #print("Decoder self attention")
        a_output, _ = self.self_attention(Q=x, K=x, V=x, relative_ids=dec_relative_ids1, mask=trg_mask)

        # Add and norm
        a_output = self.dropout(a_output)
        x = self.norm1(x + a_output)

        if enc_output is not None:
            # Compute encoder-decoder cross attention
            if self.check == "rel-e" or self.check == "rel-b" or self.check == "rel-eb":
                a_output, _ = self.cross_attention(Q=x, K=enc_output, V=enc_output, mask=src_mask)
            else:
                a_output, _ = self.cross_attention(Q=x, K=enc_output, V=enc_output, relative_ids=dec2enc_relative_ids, mask=src_mask)

            # Add and norm
            a_output = self.dropout(a_output)
            x = self.norm2(x + a_output)
        
        # Feed forward network
        ff_output = self.ffn(x)

        # Add and norm
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_hidden, dropout) for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x 
    
class ExtendedEncoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden, dropout, attention, shared_weights=False):
        super(ExtendedEncoder, self).__init__()
        if not shared_weights:
            self.layers = nn.ModuleList([ExtendedEncoderLayer(d_model, num_heads, ffn_hidden, dropout, attention) for _ in range(n_layers)])
        else:
            layer = ExtendedEncoderLayer(d_model, num_heads, ffn_hidden, dropout, attention)
            self.layers = nn.ModuleList([layer for _ in range(n_layers)])

    def forward(self, x, relative_ids, src_mask=None):
        for layer in self.layers:
            x = layer(x, relative_ids, src_mask)
        
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, ffn_hidden, dropout) for _ in range(n_layers)])
    
    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        return trg

class ExtendedDecoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, ffn_hidden, dropout, attention, shared_weights=False):
        super(ExtendedDecoder, self).__init__()
        if not shared_weights:
            self.layers = nn.ModuleList([ExtendedDecoderLayer(d_model, num_heads, ffn_hidden, dropout, attention) for _ in range(n_layers)])
        else:
            layer = ExtendedDecoderLayer(d_model, num_heads, ffn_hidden, dropout, attention)
            self.layers = nn.ModuleList([layer for _ in range(n_layers)])

    def forward(self, trg, enc_src, dec_relative_ids1, dec2enc_relative_ids, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg = layer(trg, enc_src, dec_relative_ids1, dec2enc_relative_ids, src_mask, trg_mask)
        
        return trg


# l: number of encoder/decoder layers
# d: dimensionality of token embeddings
# f: intermediate dimensionality used by the feed-forward sublayer
# h: number of attention-heads in the attention sublayers
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout=0.0):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d)
        torch.nn.init.uniform_(self.encoder_embedding.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d)
        torch.nn.init.uniform_(self.decoder_embedding.weight, -0.05, 0.05) # Embedding layer initialized with U(-0.05, 0.05)
        self.positional_encoding_enc = Encoding.AbsolutePositionalEncoding(d, max_seq_length_enc)
        self.positional_encoding_dec = Encoding.AbsolutePositionalEncoding(d, max_seq_length_dec)

        self.encoder = Encoder(l, d, h, f, dropout)
        self.decoder = Decoder(l, d, h, f, dropout)

        self.fc = nn.Linear(d, trg_vocab_size)
        torch.nn.init.xavier_uniform_(self.fc.weight) # Dense layer initialized with Glorot initializer
        self.dropout = nn.Dropout(dropout)

    def create_padding_mask(self, seq, device='cuda'):
        # Check where the sequence equals 0 (padding tokens), and cast to float
        mask = (seq == 0).float()  # Shape: [batch_size, seq_len]

        # Add extra dimensions to match the shape needed for attention mechanisms
        return mask[:, None, None, :].to(device)  # Shape: [batch_size, 1, 1, seq_len]
    
    def create_look_ahead_mask(self, size, device='cuda'):
        # Create a matrix with ones above the diagonal and zeros on or below the diagonal
        mask = torch.triu(torch.ones((size, size)), diagonal=1)  # Shape: [size, size]
        return mask.to(device)  # (seq_len, seq_len)

    def forward(self, src, trg):
        src_embedded = self.dropout(self.positional_encoding_enc(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding_dec(self.decoder_embedding(trg)))

        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask 
        enc_output = self.encoder(enc_output, src_mask)

        dec_output = tgt_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.size(1))
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.max(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder. It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, src_mask, trg_mask)
        output = torch.softmax(self.fc(dec_output), dim=-1)

        return output


class ExtendedTransformer1(Transformer):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout=0.0, attention="rel-eb"):
        super(ExtendedTransformer1, self).__init__(src_vocab_size, trg_vocab_size, d, h, l ,f, max_seq_length_enc, max_seq_length_dec, dropout)
        self.encoder = ExtendedEncoder(l, d, h, f, dropout, attention)
        self.decoder = ExtendedDecoder(l, d, h, f, dropout, attention)
    
    def forward(self, src, trg, enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids):
        src_embedded = self.dropout(self.positional_encoding_enc(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding_dec(self.decoder_embedding(trg)))

        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask
        enc_output = self.encoder(enc_output, enc_relative_ids, src_mask)

        dec_output = tgt_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.size(1))
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.max(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder. It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, dec_relative_ids1, dec2enc_relative_ids, trg_mask, src_mask)
        output = torch.softmax(self.fc(dec_output), dim=-1)

        return output


class CopyDecoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(CopyDecoder, self).__init__()
        self.attention = Attention.MultiHeadAttention(d_model, num_heads, dropout) # Scaled Dot Product Attention

        self.fcQ = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform_(self.fcQ.weight) # Dense layer initialized with Glorot initializer
        self.fcw = nn.Linear(d_model, 1)
        torch.nn.init.xavier_uniform_(self.fcw.weight) # Dense layer initialized with Glorot initializer
    
    # p1 is the Transformer output without the Copy Decoder
    def forward(self, src_vocab_size, dec_output, enc_output, src, p1):
        src_one_hot = F.one_hot(src, src_vocab_size)
        src_one_hot = src_one_hot.float()
        #print(src_one_hot.shape)

        copy_query = self.fcQ(dec_output)
        a_output, _ = self.attention(Q=copy_query, K=enc_output, V=src_one_hot, copy=True) # Authors do not use any mask
        p2 = torch.softmax(a_output, dim=-1)

        w = torch.sigmoid(self.fcw(dec_output))
        output = (p2 * w) + ((1 - w) * p1)

        return output

class ExtendedStdTransformer2(Transformer):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout=0.0):
        super(ExtendedStdTransformer2, self).__init__(src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout)
        self.copy_decoder = CopyDecoder(d, h, dropout)

    def forward(self, src, trg, src_vocab_size):
        src_embedded = self.dropout(self.positional_encoding_enc(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding_dec(self.decoder_embedding(trg)))

        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask
        enc_output = self.encoder(enc_output, src_mask)

        dec_output = tgt_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.size(1))
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.max(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder. It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, src_mask, trg_mask)
        output = torch.softmax(self.fc(dec_output), dim=-1)

        copy_output = self.copy_decoder(src_vocab_size, dec_output, enc_output, src, output)

        return copy_output

class ExtendedTransformer2(ExtendedTransformer1):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout=0.0, attention="rel-eb"):
        super(ExtendedTransformer2, self).__init__(src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout, attention)
        self.copy_decoder = CopyDecoder(d, h, dropout)
    
    def forward(self, src, trg, enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids, src_vocab_size):
        src_embedded = self.dropout(self.positional_encoding_enc(self.encoder_embedding(src)))
        trg_embedded = self.dropout(self.positional_encoding_dec(self.decoder_embedding(trg)))

        enc_output = src_embedded
        src_mask = self.create_padding_mask(src) # Encoder padding mask
        enc_output = self.encoder(enc_output, enc_relative_ids, src_mask)

        dec_output = trg_embedded
        src_mask = self.create_padding_mask(src) # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.

        look_ahead_mask = self.create_look_ahead_mask(trg.size(1))
        dec_trg_padding_mask = self.create_padding_mask(trg)
        trg_mask = torch.max(look_ahead_mask, dec_trg_padding_mask) # Used in the 1st attention block in the decoder. It is used to pad and mask future tokens in the input received by the decoder.
        
        dec_output = self.decoder(dec_output, enc_output, dec_relative_ids1, dec2enc_relative_ids, trg_mask, src_mask)
        output = torch.softmax(self.fc(dec_output), dim=-1)

        copy_output = self.copy_decoder(src_vocab_size, dec_output, enc_output, src, output)

        return copy_output


class ExtendedTransformer4(ExtendedTransformer2):
    def __init__(self, src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout=0.0, attention="rel-eb", shared_weights=True):
        super(ExtendedTransformer4, self).__init__(src_vocab_size, trg_vocab_size, d, h, l, f, max_seq_length_enc, max_seq_length_dec, dropout, attention)
        self.encoder = ExtendedEncoder(l, d, h, f, dropout, attention, shared_weights)
        self.decoder = ExtendedDecoder(l, d, h, f, dropout, attention, shared_weights)