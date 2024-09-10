import math

import torch
from transformers.models.bloom.modeling_flax_bloom import build_alibi_tensor



def generate_alibi_bias(batch_size, seq_length, num_heads, dtype=torch.float32):
    """
    Generate ALiBi bias tensor using transformers implementation.

    Args:
    batch_size (int): Batch size
    seq_length (int): Sequence length
    num_heads (int): Number of attention heads
    dtype (torch.dtype): Data type of the output tensor

    Returns:
    torch.Tensor: ALiBi bias tensor
    """
    # 创建一个全为1的attention mask
    attention_mask = torch.ones(batch_size, seq_length, dtype=dtype)

    # 调用build_alibi_tensor函数
    alibi = build_alibi_tensor(attention_mask, num_heads, dtype=dtype)

    return alibi


# 使用示例
batch_size = 2
seq_length = 10
num_heads = 8

alibi_bias = generate_alibi_bias(batch_size, seq_length, num_heads)
print(f"ALiBi bias shape: {alibi_bias.shape}")
print(f"ALiBi bias (partial):\n{alibi_bias}")
