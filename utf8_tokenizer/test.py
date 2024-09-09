from utf8_tokenizer import UTF8Tokenizer

if __name__ == "__main__":
    # Example usage
    tokenizer = UTF8Tokenizer()  # You need to provide your utf8_tokenizer here
    t = tokenizer.encode('arithmetic_training_data.jsonl')
    print(t)
