# 加载最佳模型
import os

import torch
from torch.utils.data import DataLoader

from config import TidalConfig
from data.prepare_text import TidalTextDataset
from model import TidalTransformer
from tokenizer import MixedTokenizer


def show_model_parameters(model):
    print(f"Model params:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Elements: {param.numel()}")


def decode_text(generated, start_pos, tokenizer):
    old = tokenizer.decode(generated[0][:start_pos])
    new = tokenizer.u8_decode(generated[0][start_pos:])[::-1]
    print(11, generated[0][:start_pos], old)
    print(22, generated[0][start_pos:], new)
    return old + new


def generate_text(model, tokenizer, prompt, max_new_tokens):
    current = prompt
    inited = False
    origin_start_pos = 0
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(tokenizer.encode(current)).to(next(model.parameters()).device)
        start_pos = input_ids.size(0)
        print(3333, input_ids, start_pos)
        if not inited:
            origin_start_pos = start_pos
            inited = True
        if start_pos - origin_start_pos > max_new_tokens:
            break
        generated = model.generate(input_ids, start_pos, max_new_tokens * 4, tokenizer.eob_token_id,
                                   tokenizer.eos_token_id)
        print(4444, generated)
        current = decode_text(generated, start_pos, tokenizer)
        if generated[0][-1] == tokenizer.eos_token_id:
            break
    return current


def valid_generate(prompt="54663 + 132 ="):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    tokenizer = MixedTokenizer()

    config = TidalConfig()
    config.vocab_size = tokenizer.vocab_size
    config.output_vocab_size = tokenizer.u8_vocab_size
    config.dropout = 0

    model = TidalTransformer(config)
    # 最后加载模型权重
    model.load_state_dict(torch.load('best_model.pth'))
    # model.load_state_dict(torch.load('model_step_10000.pth'))
    model.to(device)
    model.eval()
    # show_model_parameters(model)
    # 生成文本示例
    res_text = generate_text(model, tokenizer, prompt, max_new_tokens=10)
    print(f"Generated text: {res_text}")


if __name__ == "__main__":
    valid_generate()
