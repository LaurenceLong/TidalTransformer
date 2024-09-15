import math

import torch

from models.model import TidalTransformer


# 假设我们有一个简化版的 TidalConfig
class TidalConfig:
    def __init__(self):
        self.hidden_size = 64
        self.num_heads = 4
        self.vocab_size = 1000
        self.output_vocab_size = 1000
        self.num_layers = 2
        self.dropout = 0.1


# 假设 TransformerBlock 和其他必要的函数已经定义

def test_tidal_transformer_loss():
    # 初始化配置和模型
    cfg = TidalConfig()
    model = TidalTransformer(cfg)

    # 创建一个简单的输入序列
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(1, cfg.vocab_size, (batch_size, seq_len))  # 避免使用 pad_token_id (0)
    start_pos = torch.tensor([3])  # 不同的 start_pos 用于测试

    # 创建一个完全正确的预测
    perfect_logits = torch.full((batch_size, seq_len, cfg.vocab_size), -float('inf'))
    for i in range(batch_size):
        for j in range(start_pos[i], seq_len):
            perfect_logits[i, j, input_ids[i, min(j + 1, seq_len - 1)]] = 0  # 给正确的下一个 token 一个 logit 值 0

    # 打印一些调试信息
    print("Input IDs:")
    print(input_ids)
    print(perfect_logits)
    print("Perfect Logits shape:", perfect_logits.shape)
    print("Start positions:", start_pos)

    # 计算损失
    loss = model.compute_loss(perfect_logits, input_ids, start_pos)

    print(f"Loss with perfect prediction: {loss.item()}")
    assert math.isclose(loss.item(), 0, abs_tol=1e-6), f"Loss should be close to 0, but got {loss.item()}"

    # 测试一个不完美的预测
    imperfect_logits = perfect_logits.clone()
    imperfect_logits[0, -2, input_ids[0, -1]] = -float('inf')  # 将倒数第二个 token 的预测改为不正确
    imperfect_logits[0, -2, (input_ids[0, -1] + 1) % cfg.vocab_size] = 0  # 给一个错误的 token 较高的概率
    imperfect_loss = model.compute_loss(imperfect_logits, input_ids, start_pos)

    print(f"Loss with imperfect prediction: {imperfect_loss.item()}")
    assert imperfect_loss.item() > 0, f"Loss should be greater than 0 for imperfect prediction, but got {imperfect_loss.item()}"

    print("All tests passed!")


# 运行测试
test_tidal_transformer_loss()
