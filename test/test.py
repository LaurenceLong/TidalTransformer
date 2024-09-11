import torch
import numpy as np


class PretrainDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        y = np.array(sample[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(y)


# 示例用法
if __name__ == "__main__":
    # 假设我们有一些预处理好的数据
    sample_data = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500]
    ]

    # 创建数据集实例
    dataset = PretrainDataset(sample_data)

    # 获取数据集的长度
    print(f"Dataset length: {len(dataset)}")

    # 使用__getitem__方法获取一个样本
    index = 1  # 我们将获取第二个样本
    X, y = dataset[index]

    print(f"Input index: {index}")
    print(f"X (input sequence): {X}")
    print(f"y (target sequence): {y}")

    # 展示X和y的关系
    print("\nRelationship between X and y:")
    for i in range(len(X)):
        print(f"X[{i}] = {X[i]}, y[{i}] = {y[i]}")
