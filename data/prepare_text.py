import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from data.nested_list_index import NestedListIndex
from tokenizer import MixedTokenizer


class TidalTextDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_seq_len=256, cache_dir='.cache'):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data = []
        self.data_lengths = []
        self.data_idx = []

        for path in data_paths:
            cache_file = os.path.join(cache_dir, f"{os.path.basename(path)}.parquet")
            if not os.path.exists(cache_file):
                os.makedirs(cache_dir, exist_ok=True)
                self.process_file(path, tokenizer, cache_file)

            table = pq.read_table(cache_file)
            self.data.append(table)
            self.data_lengths.append(len(table.column('lengths').to_pylist()))

        self.total_length = sum(self.data_lengths)
        self.nested_list_index = NestedListIndex(self.data_lengths)

    @staticmethod
    def process_file(input_file, tokenizer: MixedTokenizer, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        tokenized_lines = []
        lengths = []
        for line in lines:
            tokens = tokenizer.encode(line.strip())
            visited = []
            for i in range(len(tokens) - 1):
                visited.append(tokens[i])
                token_text = tokenizer.decode([tokens[i + 1]])
                token_to_u8 = tokenizer.u8_encode(token_text)
                token_to_u8.reverse()
                data = visited + [tokenizer.bob_token_id] + token_to_u8 + [tokenizer.eob_token_id, len(visited)]
                tokenized_lines.append(data)
                lengths.append(len(data))

        table = pa.Table.from_arrays(
            [pa.array(tokenized_lines, type=pa.list_(pa.uint16())), lengths],
            names=['tokens', 'lengths']
        )
        pq.write_table(table, output_file)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        data_index, elem_idx = self.nested_list_index.find_list_index(idx)
        table = self.data[data_index]
        tokens_column = table.column('tokens')
        sample = tokens_column[elem_idx].as_py()

        if len(sample) > self.max_seq_len:
            sample = sample[-self.max_seq_len:]

        src = torch.tensor(sample[:-1], dtype=torch.long)  # 去掉最后一个元素（Z）
        start_pos = sample[-1]

        # 如果序列长度小于 max_seq_len，进行填充
        if len(src) < self.max_seq_len:
            padding = torch.full((self.max_seq_len - len(src),), 0, dtype=torch.long)
            src = torch.cat([src, padding])

        return src, start_pos


def custom_collate_fn(batch):
    srcs, start_poses = zip(*batch)

    # 将所有 src 堆叠成一个张量
    src_stack = torch.stack(srcs)

    # 将 start_pos 转换为张量
    start_pos_tensor = torch.tensor(start_poses, dtype=torch.long)

    return src_stack, start_pos_tensor
