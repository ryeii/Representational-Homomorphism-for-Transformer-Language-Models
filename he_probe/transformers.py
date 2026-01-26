# transformers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

# ------------------------
# Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ------------------------
# Transformer Blocks
# ------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (T, B, D)
        h = x
        x2, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(h + self.dropout(x2))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x

# ------------------------
# Decoder-Only Transformer
# ------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, d_ff=256, max_len=50, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(self, x):
        # x: (B, T)
        x = self.token_emb(x)          # (B, T, D)
        x = self.pos_emb(x)
        x = x.transpose(0,1)           # (T, B, D) for nn.MultiheadAttention
        hidden_states = []
        for layer in self.layers:
            x = layer(x, attn_mask=self._causal_mask(x.size(0), x.device))
            hidden_states.append(x.transpose(0,1))  # store (B, T, D)
        x = self.ln_final(x)
        out = self.head(x.transpose(0,1))  # (B, T, V)
        return out, hidden_states

    def _causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def get_hidden_states(self, x):
        """Return a list of hidden states (B, T, D) per layer"""
        self.eval()
        with torch.no_grad():
            _, hidden_states = self.forward(x)
        return hidden_states

# ------------------------
# Encoder-Decoder Transformer
# ------------------------
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, d_model=128, n_layers_enc=4, n_layers_dec=4,
                 n_heads=4, d_ff=256, max_len=50, dropout=0.1):
        super().__init__()
        # embeddings
        self.src_emb = nn.Embedding(vocab_size_src, d_model)
        self.tgt_emb = nn.Embedding(vocab_size_tgt, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.pos_dec = PositionalEncoding(d_model, max_len)

        # encoder
        self.encoder_layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_enc)])
        self.encoder_ln = nn.LayerNorm(d_model)

        # decoder
        self.decoder_layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_dec)])
        self.decoder_ln = nn.LayerNorm(d_model)

        # output head
        self.head = nn.Linear(d_model, vocab_size_tgt)
        self.d_model = d_model

    def forward(self, src, tgt):
        # src: (B, T_src), tgt: (B, T_tgt)
        src_h = self.src_emb(src)
        src_h = self.pos_enc(src_h).transpose(0,1)  # (T, B, D)
        for layer in self.encoder_layers:
            src_h = layer(src_h)
        src_h = self.encoder_ln(src_h)

        tgt_h = self.tgt_emb(tgt)
        tgt_h = self.pos_dec(tgt_h).transpose(0,1)
        for layer in self.decoder_layers:
            tgt_h = layer(tgt_h, attn_mask=self._causal_mask(tgt_h.size(0), tgt_h.device))
        tgt_h = self.decoder_ln(tgt_h)
        out = self.head(tgt_h.transpose(0,1))
        return out

    def _causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def get_hidden_states(self, src, tgt):
        """Return decoder hidden states list (B, T, D) per layer"""
        self.eval()
        with torch.no_grad():
            src_h = self.src_emb(src)
            src_h = self.pos_enc(src_h).transpose(0,1)
            for layer in self.encoder_layers:
                src_h = layer(src_h)
            src_h = self.encoder_ln(src_h)

            tgt_h = self.tgt_emb(tgt)
            tgt_h = self.pos_dec(tgt_h).transpose(0,1)
            hidden_states = []
            for layer in self.decoder_layers:
                tgt_h = layer(tgt_h, attn_mask=self._causal_mask(tgt_h.size(0), tgt_h.device))
                hidden_states.append(tgt_h.transpose(0,1))
        return hidden_states
