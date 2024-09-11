import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        answer = item['answer']

        # 将 prompt 和 answer 编码并连接
        input_ids = self.tokenizer.encode(prompt + answer, truncation=True, max_seq_len=self.max_seq_len)

        # 创建 attention mask
        attention_mask = [1] * len(input_ids)

        # 找到 answer 开始的位置
        answer_start = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        # 创建 labels，将 prompt 部分设为 -100（PyTorch 中用于忽略的值）
        labels = [-100] * answer_start + input_ids[answer_start:]

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
