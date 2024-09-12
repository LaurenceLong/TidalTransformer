import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            mask = mask.expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
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

        return self.W_o(x)


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

    def __init__(self, cfg: TidalConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.hidden_size * 4, cfg.dropout) for _ in
             range(cfg.num_layers)]
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.output_vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.pad_token_id = 0

    def generate_custom_mask(self, seq_len, start_pos):
        batch_size = start_pos.size(0)
        device = start_pos.device

        # 创建基础掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 创建行和列索引
        row_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)
        col_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1)

        # 创建 start_pos 掩码
        start_pos_mask = (row_indices < start_pos.unsqueeze(1).unsqueeze(2)) | (
                    col_indices < start_pos.unsqueeze(1).unsqueeze(2))

        # 合并掩码
        final_mask = mask | start_pos_mask

        # 转换为浮点数
        return final_mask.float()

    def forward(self, x, start_pos, attention_mask=None):
        batch_size, seq_len = x.size()

        # Embedding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)

        # Generate custom attention mask
        if attention_mask is None:
            attention_mask = self.generate_custom_mask(seq_len, start_pos).to(x.device)

        # Add ALIBI positional encoding
        alibi = build_alibi_tensor(attention_mask, self.num_heads, start_pos, x.dtype)

        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask, alibi)

        # Output layer
        logits = self.fc(x)

        # 处理每个批次样本的 start_pos
        batch_size, seq_len, vocab_size = logits.shape
        masked_logits = torch.zeros_like(logits)

        for i in range(batch_size):
            masked_logits[i, start_pos[i]:, :] = logits[i, start_pos[i]:, :]

        return masked_logits

    def compute_loss(self, logits, targets, start_pos):
        batch_size, seq_len, vocab_size = logits.shape

        # 创建一个掩码来选择有效的位置
        mask = torch.arange(seq_len, device=logits.device).unsqueeze(0) >= start_pos.unsqueeze(1)

        # 将 logits 和 targets 展平，只保留有效的位置
        flat_logits = logits[mask]
        flat_targets = targets[mask]

        # 计算损失
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.pad_token_id)

        return loss

    def generate(self, x, start_pos,max_new_tokens, eob_token_id):
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批量维度
        if isinstance(start_pos, int):
            start_pos = torch.tensor([start_pos], device=x.device)
        if start_pos.dim() == 0:
            start_pos = start_pos.unsqueeze(0)  # 添加批量维度
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(x, start_pos)
                next_token = logits[:, -1, :].argmax(dim=-1)
                x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
                if next_token == eob_token_id:
                    break
        return x
