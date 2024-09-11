import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


def process_file(file_path, tokenizer, output_path):
    doc_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line)
            text = line_data['text']
            text_id = tokenizer.encode(text, add_special_tokens=False)
            text_id.append(tokenizer.eos_token_id)
            if len(text_id) > 5:
                doc_ids.extend(text_id)

    arr = np.array(doc_ids, dtype=np.uint16)
    with open(output_path, 'wb') as f:
        f.write(arr.tobytes())


class TextDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_seq_len, cache_dir='.cache', use_memmap=True):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.use_memmap = use_memmap

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data = []
        self.data_lengths = []

        for path in data_paths:
            cache_file = os.path.join(cache_dir, f"{os.path.basename(path)}.bin")
            if not os.path.exists(cache_file):
                os.makedirs(cache_dir, exist_ok=True)
                process_file(path, tokenizer, cache_file)

            if use_memmap:
                data = np.memmap(cache_file, dtype=np.uint16, mode='r')
            else:
                with open(cache_file, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)

            self.data.append(data)
            self.data_lengths.append(len(data))

        self.total_length = sum(self.data_lengths)

    def __len__(self):
        return self.total_length - self.max_seq_len * len(self.data)

    def __getitem__(self, idx):
        # Find which file the index belongs to
        for i, length in enumerate(self.data_lengths):
            if idx < length - self.max_seq_len:
                data = self.data[i]
                break
            idx -= (length - self.max_seq_len)

        sample = data[idx:idx + self.max_seq_len + 1]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)
