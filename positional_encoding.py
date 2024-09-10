import numpy as np
import torch


def generate_alibi_matrix(seq_length, bob_index):
    distances = np.array([i - bob_index if i >= bob_index else seq_length - bob_index - 1 + abs(i - bob_index) for i in
                          range(seq_length)])
    return distances[:, np.newaxis] - distances

def  build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    pass

if __name__ == "__main__":
    # 示例使用
    seq_length = 10  # 总序列长度
    n = 5  # <bob>的位置（从0开始计数）
    alibi_matrix = generate_alibi_matrix(seq_length, n)
    print(alibi_matrix)
