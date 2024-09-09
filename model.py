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
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.intermediate = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
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
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_alibi_bias(max_seq_length, num_heads)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_alibi_bias(self, max_seq_length, num_heads):
        slopes = torch.Tensor([2 ** (-8 * i / num_heads) for i in range(num_heads)])
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_length).unsqueeze(0).unsqueeze(0).expand(
            num_heads, -1, -1)
        alibi = alibi.view(1, num_heads, 1, max_seq_length)
        return alibi

    def forward(self, x, attention_mask=None):
        seq_length = x.size(1)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)

        # Add ALIBI positional encoding
        alibi = self.positional_encoding[:, :, :, :seq_length].expand(batch_size, -1, seq_length, -1).to(x.device)

        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output layer
        output = self.fc(x)
        return output


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


# Usage example
vocab_size = 30000  # Example vocabulary size
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 1024
dropout = 0.1

model = TidalTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)


# Example input processing and forward pass
# Note: You would need to implement or use an actual tokenizer
class DummyTokenizer:
    def __init__(self):
        self.bob_token_id = 1  # Example token ID for <bob>
        self.eob_token_id = 2  # Example token ID for <eob>

    def encode(self, text):
        # This is a dummy implementation. Replace with actual tokenization logic.
        return [ord(c) for c in text]


tokenizer = DummyTokenizer()
input_text = "Math Equation"
block_size = 4
input_ids = process_input(input_text, tokenizer, block_size)
attention_mask = create_block_attention_mask(input_ids.size(1), block_size)

output = model(input_ids, attention_mask)
print(output.shape)  # Should print: torch.Size([1, seq_length, vocab_size])
