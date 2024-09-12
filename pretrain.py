from datetime import datetime
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import TidalConfig
from data.prepare_text import TidalTextDataset
from model import TidalTransformer
from tokenizer import MixedTokenizer


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_log_{timestamp}.log"
log = get_logger(log_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def train(model, train_dataloader, val_dataloader, config):
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas,
                            weight_decay=config.weight_decay)

    best_val_loss = float('inf')
    step = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        with logging_redirect_tqdm():
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=True):
                input_ids, start_pos = batch
                input_ids, start_pos = input_ids.to(device), start_pos.to(device)

                optimizer.zero_grad()

                logits = model(input_ids, start_pos)
                loss = model.compute_loss(logits, input_ids, start_pos)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                step += 1

                if step % config.log_interval == 0:
                    log.info(f"Step {step}, Loss: {loss.item():.4f}")

                if step % config.save_interval == 0:
                    torch.save(model.state_dict(), f'model_step_{step}.pth')
                    log.info(f"Saved model at step {step}")

            avg_train_loss = total_loss / len(train_dataloader)
            log.info(f"Epoch {epoch + 1}, Average train loss: {avg_train_loss:.4f}")

            if (epoch + 1) % config.eval_interval == 0:
                val_loss = evaluate(model, val_dataloader, config)
                log.info(f"Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    log.info("Saved best model.")


def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= config.eval_iters:
                break
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            start_pos = 0

            logits = model(input_ids, start_pos, attention_mask)
            loss = model.compute_loss(logits, input_ids, start_pos)

            total_loss += loss.item()

    return total_loss / min(config.eval_iters, len(dataloader))


def generate_text(model, tokenizer, prompt, max_new_tokens, config):
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    start_pos = input_ids.size(1)

    generated = model.generate(input_ids, start_pos, max_new_tokens)

    return tokenizer.decode(generated[0])


if __name__ == "__main__":
    # 配置
    config = TidalConfig()

    # 加载tokenizer
    tokenizer = MixedTokenizer()

    # 更新配置中的词汇表大小
    config.vocab_size = tokenizer.vocab_size
    config.output_vocab_size = tokenizer.u8_vocab_size

    # 准备数据
    # 这里您需要准备自己的文本数据
    train_ds = [r'D:\work\TidalTransformer\data\arithmetic_data.text']
    val_ds = [r'D:\work\TidalTransformer\data\arithmetic_data.text']
    train_dataset = TidalTextDataset(train_ds, tokenizer, config.max_seq_len)
    val_dataset = TidalTextDataset(val_ds, tokenizer, config.max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 初始化模型
    model = TidalTransformer(config)

    # 训练模型
    train(model, train_dataloader, val_dataloader, config)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 生成文本示例
    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=50, config=config)
    log.info(f"Generated text: {generated_text}")
