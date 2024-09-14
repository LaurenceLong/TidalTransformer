import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TidalConfig
from positional_encoding import build_alibi_tensor, generate_casual_mask


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, alibi=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if alibi is not None:
            scores += alibi

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, alibi=None):
        # Pre-norm for attention
        normed_x = self.attention_norm(x)
        attention_output = self.attention(normed_x, attention_mask, alibi)
        x = x + self.dropout(attention_output)

        # Pre-norm for feed-forward
        normed_x = self.ffn_norm(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)

        return x


class TidalTransformer(nn.Module):

    def __init__(self, cfg: TidalConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.hidden_size * 4, cfg.dropout, cfg.layer_norm_eps)
             for _ in range(cfg.num_layers)]
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.output_vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.pad_token_id = 0

    def forward(self, input_ids, start_pos, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)
        # Generate custom attention mask
        if attention_mask is None:
            attention_mask = generate_casual_mask(batch_size, self.num_heads, seq_len).to(x.device)
        # Add ALIBI positional encoding
        alibi = build_alibi_tensor(attention_mask, start_pos, x.dtype)
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask, alibi)
        # Output layer
        logits = self.fc(x)
        # 使用高效的张量操作来处理 masked_logits
        batch_size, seq_len, vocab_size = logits.shape
        seq_indices = torch.arange(seq_len, device=logits.device).unsqueeze(0)
        mask = seq_indices >= start_pos.unsqueeze(1)
        masked_logits = logits.masked_fill(~mask.unsqueeze(-1), 0)
        return masked_logits

    def compute_loss(self, logits, input_ids, start_pos):
        batch_size, seq_len, vocab_size = logits.shape
        # 创建目标序列：将输入向右移动一位
        targets = input_ids[:, 1:]  # 从第二个位置开始
        targets = F.pad(targets, (0, 1), value=self.pad_token_id)  # 在末尾添加填充
        # 使用更高效的张量操作创建loss_mask
        seq_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        loss_mask = seq_indices >= start_pos.unsqueeze(1)
        # 应用掩码
        valid_logits = logits[loss_mask]
        valid_targets = targets[loss_mask]
        # 计算交叉熵损失
        loss = F.cross_entropy(valid_logits, valid_targets, ignore_index=self.pad_token_id)
        return loss

    def generate(self, x, start_pos, max_new_chars, eob_token_id, eos_token_id, temperature=1.0, top_k=0, top_p=1.0):
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批量维度
        if isinstance(start_pos, int):
            start_pos = torch.tensor([start_pos], device=x.device)

        batch_size = x.size(0)
        with torch.no_grad():
            for _ in range(max_new_chars):
                logits = self(x, start_pos)
                next_token_logits = logits[:, -1, :] / temperature

                for i in range(batch_size):
                    if top_k > 0:
                        indices_to_remove = next_token_logits[i] < torch.topk(next_token_logits[i], top_k)[0][-1]
                        next_token_logits[i][indices_to_remove] = -float('Inf')

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits[i], descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[i][indices_to_remove] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                x = torch.cat([x, next_token.unsqueeze(1)], dim=1)

                if (next_token == eob_token_id).all() or (next_token == eos_token_id).all():
                    break
        return x
