import numpy as np
import torch
import math


def get_slopes(n_heads):
    def get_slopes_power_of_2(n_heads):
        start = (2 ** (-2 ** -(math.log2(n_heads) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n_heads)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n_heads - closest_power_of_2]


def example_alibi_distance_matrix(seq_length, start_pos, sqrt=False):
    if not sqrt:
        distances = np.array([i - start_pos if i >= start_pos else
                              seq_length - start_pos - 1 + abs(i - start_pos) for i in range(seq_length)])
    else:
        distances = np.array([round(np.sqrt(i - start_pos), 2) if i >= start_pos else
                              round(np.sqrt(seq_length - start_pos - 1), 2) + abs(i - start_pos) for i in
                              range(seq_length)])
    return distances[:, np.newaxis] - distances


def generate_tidal_positions(seq_length, start_pos):
    device = start_pos.device  # 获取start_pos的设备
    indices = torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(0)
    start_pos = start_pos.unsqueeze(1).float()
    distances = torch.where(
        indices >= start_pos,
        seq_length - start_pos - 1 - ((indices - start_pos) / (seq_length - start_pos + 0.618)),
        seq_length - start_pos - 1 - torch.abs(indices - start_pos)
    )
    return distances


def generate_tidal_rev_positions(seq_length, start_pos):
    device = start_pos.device  # 获取start_pos的设备
    indices = torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(0)
    start_pos = start_pos.unsqueeze(1).float()

    # distances = torch.where(
    #     indices >= start_pos,
    #     indices - start_pos,
    #     (seq_length - start_pos - 1) + torch.abs(indices - start_pos)
    # )
    distances = torch.where(
        indices >= start_pos,
        (indices - start_pos) / (seq_length - start_pos + 0.382),
        torch.abs(indices - start_pos)
    )
    return distances


def generate_tidal_rev_distance_matrix(seq_length, start_pos):
    distances = generate_tidal_rev_positions(seq_length, start_pos)
    return distances.unsqueeze(2) - distances.unsqueeze(1)


def build_alibi_tensor(attention_mask: torch.Tensor, start_pos: torch.Tensor,
                       dtype: torch.dtype) -> torch.Tensor:
    batch_size, num_heads, seq_length, _ = attention_mask.shape
    # 生成距离矩阵
    distances = generate_tidal_rev_distance_matrix(seq_length, start_pos)
    # 获取每个头的斜率
    slopes = torch.tensor(get_slopes(num_heads), dtype=dtype, device=attention_mask.device)
    # 计算alibi偏置
    alibi = slopes.view(1, num_heads, 1, 1) * distances.unsqueeze(1)
    return alibi


def generate_casual_mask(batch_size, num_heads, seq_len):
    # 创建基础掩码（下三角矩阵）
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # 扩展掩码到批次+ num_heads维度
    mask = mask.unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    return mask


def test_build_alibi_tensor():
    # 测试用例1：基本功能测试
    batch_size = 1
    num_heads = 1
    seq_len = 9
    pos = 3

    input_ids = torch.ones(batch_size, seq_len)
    start_pos = torch.tensor([pos])

    attention_mask = generate_casual_mask(batch_size, num_heads, seq_len)
    print("0000\n", attention_mask)

    print(1111, example_alibi_distance_matrix(seq_len, pos))
    print(1111, example_alibi_distance_matrix(seq_len, pos, True))

    distances = generate_tidal_rev_distance_matrix(seq_len, start_pos)
    print(2222, distances)

    print(3333, get_slopes(num_heads))

    # result = build_alibi_tensor(attention_mask, start_pos, dtype)
    # print(8888, result.shape)
    # print(9999, result)

    print(4444, generate_tidal_positions(seq_len, start_pos))
    print(5555, generate_tidal_rev_positions(seq_len, start_pos))


if __name__ == "__main__":
    # 运行测试
    test_build_alibi_tensor()
