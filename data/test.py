import config
from prepare_text import TextDataset
from tokenizer import UTF8Tokenizer, MixedTokenizer

if __name__ == "__main__":
    # Example usage
    tokenizer = MixedTokenizer()  # You need to provide your tokenizer here
    dataset = TextDataset('arithmetic_training_data.jsonl', tokenizer, max_seq_len=config.ModelConfig.max_seq_len)
    print(f"Dataset length: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample input shape: {x.shape}, Sample output shape: {y.shape}")
