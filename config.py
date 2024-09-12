from dataclasses import dataclass


@dataclass
class TidalConfig:
    # model params
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = -1
    output_vocab_size: int = -1
    max_seq_len: int = 128
    dropout: float = 0
    # training params...
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 32
    num_epochs: int = 1
    eval_interval: int = 1
    log_interval: int = 100
    save_interval: int = 10000
    eval_iters: int = 200
