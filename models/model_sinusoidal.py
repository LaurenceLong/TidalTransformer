import math

from config import TidalConfig
from models.base_model import create_sinusoidal_embeddings, TidalTransformerBase
from positional_encoding import generate_casual_mask, generate_tidal_positions


class TidalTransformer(TidalTransformerBase):

    def __init__(self, cfg: TidalConfig):
        super().__init__(cfg=cfg)

        self.pos_embedding = create_sinusoidal_embeddings(cfg.max_seq_len, cfg.hidden_size)

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
        # sinusoidal_embeddings
        positions = generate_tidal_positions(seq_length, start_pos).long()
        position_embeddings = self.pos_embedding.squeeze(0)  # 移除第一个维度，现在形状为 [max_seq_len, hidden_size]
        position_embeddings = position_embeddings[positions]  # 使用高级索引，形状变为 [batch_size, seq_len, hidden_size]
        x = x + position_embeddings
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        # Output layer
        logits = self.fc(x)
        return logits
