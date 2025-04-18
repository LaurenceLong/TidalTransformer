from dataclasses import dataclass


class InitFrom:
    scratch: int = 0
    resume: int = 1


@dataclass
class TidalConfig:
    # model params
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = -1
    char_vocab_size: int = -1
    token_vocab_size: int = -1
    max_seq_len: int = 128
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    # training params...
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 32
    num_epochs: int = 10
    log_interval: int = 100
    eval_interval: int = 500
    eval_iters: int = 50
    save_interval: int = 5000
    init_from: int = InitFrom.scratch
