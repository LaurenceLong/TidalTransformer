import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        return torch.matmul(attention_probs, value)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        query = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        x = self.dropout(token_embeddings + position_embeddings)

        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device)).unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# 使用示例
vocab_size = 30000
d_model = 768
num_heads = 12
num_layers = 12
d_ff = 3072
max_seq_length = 1024
dropout = 0.1

model = TransformerDecoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# 假设输入
input_ids = torch.randint(0, vocab_size, (2, 512))
attention_mask = torch.ones_like(input_ids)

# 前向传播
output = model(input_ids, attention_mask)
print(output.shape)  # 应该是 (2, 512, vocab_size)