import torch

from models.base_model import RotaryEmbedding, apply_rotary_pos_emb
from positional_encoding import generate_tidal_positions

# 测试参数
batch_size = 1
num_heads = 1
seq_len = 7
head_dim = 64
pos = 3

# 创建输入
input_ids = torch.ones(batch_size, seq_len)
start_pos = torch.tensor([pos])

# 初始化 RotaryEmbedding
rotary_emb = RotaryEmbedding(head_dim)

# 生成位置信息
positions = generate_tidal_positions(seq_len, start_pos)
print("Generated positions:")
print(positions)

# 生成 RoPE 嵌入
freqs_cos, freqs_sin = rotary_emb(positions)
print("\nShape of freqs_cos:", freqs_cos.shape)
print("Shape of freqs_sin:", freqs_sin.shape)

# 展开 freqs_cos 和 freqs_sin 以匹配 q 和 k 的形状
freqs_cos = freqs_cos.unsqueeze(1).expand(-1, num_heads, -1, -1)
freqs_sin = freqs_sin.unsqueeze(1).expand(-1, num_heads, -1, -1)

# 创建模拟的 q 和 k
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)

print("\nShape of q:", q.shape)
print("Shape of k:", k.shape)

# 应用旋转位置嵌入
q_rotated, k_rotated = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

print("\nShape of q_rotated:", q_rotated.shape)
print("Shape of k_rotated:", k_rotated.shape)

# 打印第一个位置的原始和旋转后的 q 值，以便观察变化
print("\nOriginal q[0, 0, 0]:", q[0, 0, 0])
print("Rotated q[0, 0, 0]:", q_rotated[0, 0, 0])

# 验证位置编码的效果
print("\nVerifying positional encoding effect:")
for i in range(seq_len):
    similarity = torch.nn.functional.cosine_similarity(q_rotated[0, 0, i], k_rotated[0, 0, i], dim=0)
    print(f"Cosine similarity between q_rotated and k_rotated at position {i}: {similarity.item():.4f}")
