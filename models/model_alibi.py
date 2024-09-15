import math

from config import TidalConfig
from models.base_model import TidalTransformerBase
from positional_encoding import generate_casual_mask, build_alibi_tensor


class TidalTransformer(TidalTransformerBase):

    def __init__(self, cfg: TidalConfig):
        super().__init__(cfg)

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
        # Add ALIBI positional encoding
        alibi = build_alibi_tensor(attention_mask, start_pos, x.dtype)
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, alibi=alibi)
        # Output layer
        logits = self.fc(x)
        return logits
