import torch
from model import TidalTransformer
from data import get_dataloader
from config import TidalConfig


def train(config, model, train_dataloader, val_dataloader):
    pass


# 实现训练循环

def evaluate(model, dataloader):
    pass

# 实现评估函数

def main():
    config = TidalConfig()
    model = TidalTransformer(config)

    # 加载数据
    train_dataloader = get_dataloader(...)
    val_dataloader = get_dataloader(...)

    # 训练模型
    train(config, model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()