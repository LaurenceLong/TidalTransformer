import numpy as np
import torch
import math


def generate_alibi_distance_matrix(seq_length, bob_idx):
    # distances = np.array(
    #     [(i - bob_idx) if i >= bob_idx else seq_length - bob_idx - 1 + abs(i - bob_idx) for i in range(seq_length)])
    # print(distances[:, np.newaxis] - distances)
    # downscale with sqrt
    distances = np.array([np.sqrt(i - bob_idx) if i >= bob_idx else
                          np.sqrt(seq_length - bob_idx) - 1 + abs(i - bob_idx) for i in range(seq_length)])
    return distances[:, np.newaxis] - distances


def get_slopes(n_heads):
    def get_slopes_power_of_2(n_heads):
        start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n_heads)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n_heads - closest_power_of_2]


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, start_pos: torch.Tensor,
                       dtype: torch.dtype) -> torch.Tensor:
    batch_size, seq_length, _ = attention_mask.shape

    # 生成距离矩阵
    distances = torch.arange(seq_length, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1)
    distances = distances.unsqueeze(-1) - distances.unsqueeze(-2)
    distances = distances.abs().float()

    # 应用 start_pos
    start_pos = start_pos.view(-1, 1, 1)
    distances = torch.where(
        (torch.arange(seq_length, device=attention_mask.device).unsqueeze(0) < start_pos) |
        (torch.arange(seq_length, device=attention_mask.device).unsqueeze(1) < start_pos),
        torch.zeros_like(distances),
        distances
    )

    # 获取每个头的斜率
    slopes = torch.tensor(get_slopes(num_heads), dtype=dtype, device=attention_mask.device)

    # 计算alibi偏置
    alibi = slopes.view(1, num_heads, 1, 1) * distances.unsqueeze(1)

    return alibi


def test():
    # 示例使用
    seq_length = 20  # 总序列长度
    bob_idx = 5  # <bob>的位置（从0开始计数）
    num_heads = 8  # 注意力头的数量
    batch_size = 2  # 批次大小

    # 生成示例注意力掩码
    attention_mask = torch.ones(batch_size, seq_length)
    print(attention_mask.size())

    # 计算alibi张量
    alibi_tensor = build_alibi_tensor(attention_mask, num_heads, bob_idx, torch.float32)

    print("Alibi distance matrix:")
    matrix = generate_alibi_distance_matrix(seq_length, bob_idx)
    for _ in matrix:
        print(_)
    print("\nAlibi tensor shape:", alibi_tensor.shape)
    print("\nAlibi tensor for the first head in the first batch:")
    print(alibi_tensor[0, 0])

    # 验证alibi偏置的特性
    print("\nVerifying alibi bias properties:")
    print("1. Symmetry: ", torch.allclose(alibi_tensor, alibi_tensor.transpose(-1, -2)))
    print("2. Zero diagonal: ", torch.allclose(alibi_tensor.diagonal(dim1=-2, dim2=-1),
                                               torch.zeros_like(alibi_tensor.diagonal(dim1=-2, dim2=-1))))
    print("3. Negative values: ", (alibi_tensor <= 0).all())

    # 检查不同头的偏置差异
    print("\nBias difference between first and last head:")
    print(alibi_tensor[0, 0] - alibi_tensor[0, -1])


if __name__ == "__main__":
    test()
