import math

import torch.nn as nn

from config import TidalConfig
from models.base_model import TidalTransformerBase, TransformerBlock, RotaryEmbedding
from positional_encoding import generate_casual_mask, generate_tidal_positions


class TidalTransformer(TidalTransformerBase):

    def __init__(self, cfg: TidalConfig):
        super().__init__(cfg=cfg)
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

        self.rotary_emb = RotaryEmbedding(cfg.hidden_size // cfg.num_heads)

    def forward(self, input_ids, start_pos, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(x)
        # Generate custom attention mask
        if attention_mask is None:
            attention_mask = generate_casual_mask(batch_size, self.num_heads, seq_len).to(x.device)
        # 添加位置编码
        batch_size, num_heads, seq_length, _ = attention_mask.shape
        # Generate position indices
        positions = generate_tidal_positions(seq_len, start_pos).to(x.device)
        positions = positions.unsqueeze(1).expand(-1, self.num_heads, -1)

        # Generate RoPE embeddings
        freqs = self.rotary_emb(positions)

        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask, freqs=freqs)
        # Output layer
        logits = self.fc(x)
        return logits
