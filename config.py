class TidalConfig:
    def __init__(self):
        self.vocab_size = 30000
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.max_position_embeddings = 512
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 10
        # 其他配置参数...