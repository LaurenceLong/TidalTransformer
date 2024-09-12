import os

import torch
from torch.utils.data import DataLoader

# 假设ParquetTextDataset类在名为parquet_dataset.py的文件中
from prepare_text import TidalTextDataset, custom_collate_fn
from tokenizer import MixedTokenizer


def test_parquet_text_dataset():
    # 创建一个临时的文本文件
    test_file = 'test_data.txt'
    texts = ["Hello world is good asdf\n", "This is a test.\n", "Python is awesome.\n"]
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(texts)

    # 设置参数
    max_seq_len = 20
    cache_dir = '.cache'

    # 使用一个简单的tokenizer
    tokenizer = MixedTokenizer()

    # 创建数据集
    dataset = TidalTextDataset(test_file, tokenizer, max_seq_len, cache_dir)

    # 检查数据集的长度
    # expected_length = sum([len(tokenizer.encode(_)) for _ in texts])
    # assert len(dataset) == expected_length, f"Expected length {expected_length}, got {len(dataset)}"

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

    # 检查批次的形状
    for X, start_pos in dataloader:
        print(1111, X.shape)
        print(X)
        print(start_pos)
        assert X.shape == (2, max_seq_len), f"Expected X shape (2, {max_seq_len}), got {X.shape}"

    # 清理
    os.remove(test_file)
    os.remove(os.path.join(cache_dir, f"{os.path.basename(test_file)}.parquet"))

    print("All tests passed!")


if __name__ == "__main__":
    test_parquet_text_dataset()
