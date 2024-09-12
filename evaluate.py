# 加载最佳模型
import torch

from config import TidalConfig
from model import TidalTransformer
from tokenizer import MixedTokenizer


def print_model_parameters(model):
    print(f"Model params:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Elements: {param.numel()}")


def generate_text(model, tokenizer, prompt, max_new_tokens):
    model.to(device)
    model.eval()

    input_ids = torch.tensor(tokenizer.encode(prompt)).to(device)
    start_pos = input_ids.size(0) - 1

    generated = model.generate(input_ids, start_pos, max_new_tokens, tokenizer.eob_token_id)
    print(1111, generated)

    return tokenizer.decode(generated[0])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    tokenizer = MixedTokenizer()

    config = TidalConfig()
    config.vocab_size = tokenizer.vocab_size
    config.output_vocab_size = tokenizer.u8_vocab_size
    model = TidalTransformer(config)
    # 最后加载模型权重
    model.load_state_dict(torch.load('best_model.pth'))
    print_model_parameters(model)
    # 生成文本示例
    prompt = "1 + 1001 ="
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=50)
    print(f"Generated text: {generated_text}")
