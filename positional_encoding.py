import numpy as np


#
def generate_alibi_matrix(seq_length, n):
    matrix = np.zeros((seq_length, seq_length))

    distances = [_ - n if _ >= n else seq_length - n - 1 + abs(_ - n) for _ in range(seq_length)]
    for i in range(seq_length):
        for j in range(seq_length):
            matrix[i, j] = abs(distances[j] - distances[i])
    return matrix


# 示例使用
seq_length = 10  # 总序列长度
n = 5  # <bob>的位置（从0开始计数）

alibi_matrix = generate_alibi_matrix(seq_length, n)
print(alibi_matrix)


def generate_alibi_matrix(seq_length, n):
    distances = np.array([_ - n if _ >= n else seq_length - n - 1 + abs(_ - n) for _ in range(seq_length)])
    return np.abs(distances[:, np.newaxis] - distances)


# 示例使用
seq_length = 10  # 总序列长度
n = 5  # <bob>的位置（从0开始计数）

alibi_matrix = generate_alibi_matrix(seq_length, n)
print(alibi_matrix)
