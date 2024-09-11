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
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, alibi=None):
        attention_output = self.attention(hidden_states, attention_mask, alibi)
        attention_output = self.dropout(attention_output)
        hidden_states = self.layernorm1(hidden_states + attention_output)

        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.dropout(intermediate_output)
        layer_output = self.layernorm2(hidden_states + layer_output)

        return layer_output


class TidalTransformer(nn.Module):
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

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        # Embedding
        x = self.embedding(tokens) * math.sqrt(self.hidden_size)
        x = self.dropout(x)

        # Add ALIBI positional encoding
        alibi = build_alibi_tensor(attention_mask, self.num_heads, None, x.dtype)

        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask, alibi)

        # Output layer
        output = self.fc(x)
        return output

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


def create_block_attention_mask(seq_length, block_size):
    mask = torch.ones(seq_length, seq_length)
    for i in range(0, seq_length, block_size):
        end = min(i + block_size, seq_length)
        mask[i:end, i:end] = torch.tril(torch.ones(end - i, end - i))
    return mask


def process_input(text, tokenizer, block_size):
    tokens = tokenizer.encode(text)
    processed_tokens = []
    for i in range(0, len(tokens), block_size):
        block = tokens[i:i + block_size]
        if i > 0:
            processed_tokens.extend([tokenizer.bob_token_id])
            processed_tokens.extend(block[::-1])  # Reverse the block
            processed_tokens.extend([tokenizer.eob_token_id])
        else:
            processed_tokens.extend(block)
    return torch.tensor(processed_tokens).unsqueeze(0)
