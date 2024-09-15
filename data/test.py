import os

import torch
from torch.utils.data import DataLoader

from evaluate import decode_text
from data.prepare_text import TidalTextDataset, custom_collate_fn
from tokenizer import MixedTokenizer


def tst_parquet_text_dataset(tokenizer, test_file="arithmetic_test.txt"):
    # 创建一个临时的文本文件
    created = False

    test_file = 'test_data.txt'
    texts = ["Hello world is good asdf\n", "This is a test.\n", "Python is awesome.\n"]
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(texts)
        created = True

    # 设置参数
    max_seq_len = 20
    cache_dir = '.cache'

    # 创建数据集
    dataset = TidalTextDataset(test_file, tokenizer, max_seq_len, cache_dir)

    # 检查数据集的长度
    # expected_length = sum([len(tokenizer.encode(_)) for _ in texts])
    # assert len(dataset) == expected_length, f"Expected length {expected_length}, got {len(dataset)}"

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # 检查批次的形状
    for input_ids, start_pos in dataloader:
        print(input_ids)
        print(start_pos)
        print("==" * 50)
        assert input_ids.shape == (1, max_seq_len), f"Expected X shape (2, {max_seq_len}), got {input_ids.shape}"

    # 清理
    if created:
        os.remove(test_file)
        os.remove(os.path.join(cache_dir, f"{os.path.basename(test_file)}.parquet"))
    print("All tests passed!")


def valid_dataset(tokenizer, test_file='arithmetic_test.txt'):
    cwd = os.path.dirname(os.path.abspath(__file__))
    cache_dir = '.cache'
    train_ds = [os.path.join(cwd, test_file)]
    os.remove(os.path.join(cache_dir, f"{os.path.basename(test_file)}.parquet"))
    train_dataset = TidalTextDataset(train_ds, tokenizer, 20)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    for i, batch in enumerate(train_dataloader):
        input_ids, start_pos = batch
        print(7777, input_ids)
        print(8888, start_pos)
        print(9999, decode_text(input_ids, start_pos, tokenizer))
        print("==" * 50)
        if i >= 20:
            break


if __name__ == "__main__":
    tknz = MixedTokenizer()
    res = tknz.u8_decode([56, 59, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0])[::-1]
    print(1111, res)

    decode_text([[13481, 2631, 638, 2155, 728, 108, 106, 252]], 5, tknz)

    # tst_parquet_text_dataset(tknz)
    valid_dataset(tknz)
