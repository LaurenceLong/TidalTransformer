import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import TidalConfig
from positional_encoding import build_alibi_tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, alibi=None):
        d_k = query.shape[-1]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if alibi is not None:
            attention_scores += alibi
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        return torch.matmul(attention_probs, value)

    def forward(self, x, mask=None, alibi=None):
        batch_size = x.shape[0]

        query = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x = self.attention(query, key, value, mask, alibi)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, d_ff)
        self.linear2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.intermediate = FeedForward(hidden_size, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, alibi=None):
        attention_output = self.attention(hidden_states, attention_mask, alibi)
        attention_output = self.dropout(attention_output)
        hidden_states = self.layer_norm_1(hidden_states + attention_output)

        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.dropout(intermediate_output)
        layer_output = self.layer_norm_2(hidden_states + layer_output)

        return layer_output


class TidalTransformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, cfg: TidalConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.hidden_size * 4, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_loss = None

    def forward(self, x, start_pos, attention_mask=None):
        # Embedding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)

        # 将输入分为两部分：上下文和需要生成的部分
        context = x[:, :start_pos]
        to_generate = x[:, start_pos:]

        if attention_mask is None:
            attention_mask = self.generate_square_subsequent_mask(x.size(1))
        gen_mask = attention_mask[start_pos:, :start_pos]

        # Add ALIBI positional encoding
        alibi = build_alibi_tensor(attention_mask, self.num_heads, None, x.dtype)

        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask, alibi)

        # Output layer
        output = self.fc(x)
        return output


