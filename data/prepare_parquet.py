import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import MixedTokenizer


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, tokenizer):
        self.tokenizer = tokenizer
        self.table = pq.read_table(parquet_file)
        self.data = self.table.to_pandas()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 假设我们的数据有 'features' 和 'label' 列
        text = row['text']
        sample = self.tokenizer.encode(text)
        print(sample)
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)


def test():
    # 使用示例
    parquet_file = '.cache/wikitext/wiketext-2-raw-v1/test-00000-of-00001.parquet'
    dataset = ParquetDataset(parquet_file, MixedTokenizer())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历数据
    for i, (X, Y) in enumerate(dataloader):
        # 这里可以进行你的训练步骤
        print(i, X, Y)


if __name__ == '__main__':
    test()
