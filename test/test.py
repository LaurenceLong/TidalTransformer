import torch


def original_masked_logits(logits, start_pos):
    batch_size, seq_len, vocab_size = logits.shape
    masked_logits = torch.zeros_like(logits)

    for i in range(batch_size):
        masked_logits[i, start_pos[i]:, :] = logits[i, start_pos[i]:, :]

    return masked_logits


def optimized_masked_logits(logits, start_pos):
    batch_size, seq_len, vocab_size = logits.shape
    seq_indices = torch.arange(seq_len, device=logits.device).unsqueeze(0)
    mask = seq_indices >= start_pos.unsqueeze(1)
    masked_logits = logits.masked_fill(~mask.unsqueeze(-1), 0)

    return masked_logits


def test_masked_logits_equivalence():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 创建随机输入
    batch_size, seq_len, vocab_size = 4, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    start_pos = torch.randint(0, seq_len, (batch_size,))

    # 运行两个版本的代码
    original_output = original_masked_logits(logits, start_pos)
    optimized_output = optimized_masked_logits(logits, start_pos)

    # 比较结果
    are_equal = torch.allclose(original_output, optimized_output, rtol=1e-5, atol=1e-8)

    if are_equal:
        print("Test passed: Both implementations produce equivalent results.")
    else:
        print("Test failed: Implementations produce different results.")

        # 如果结果不同，打印一些调试信息
        print(f"Start positions: {start_pos}")
        print(f"Maximum absolute difference: {(original_output - optimized_output).abs().max().item()}")
        print(f"Average absolute difference: {(original_output - optimized_output).abs().mean().item()}")

        # 找出不同的位置
        diff_mask = ~torch.isclose(original_output, optimized_output, rtol=1e-5, atol=1e-8)
        diff_indices = torch.nonzero(diff_mask, as_tuple=True)
        print(f"Number of different elements: {len(diff_indices[0])}")
        if len(diff_indices[0]) > 0:
            print("Sample of differences:")
            for i in range(min(5, len(diff_indices[0]))):
                idx = tuple(index[i] for index in diff_indices)
                print(
                    f"  Position {idx}: Original: {original_output[idx].item():.6f}, Optimized: {optimized_output[idx].item():.6f}")


# 运行测试
test_masked_logits_equivalence()