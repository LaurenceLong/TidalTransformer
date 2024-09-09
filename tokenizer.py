import tiktoken

from utf8_tokenizer import UTF8Tokenizer


class MixedTokenizer:
    def __init__(self):
        self.u8_tokenizer = UTF8Tokenizer()
        self.gpt4o_tokenizer = tiktoken.get_encoding("o200k_base")  # GPT4o
        self.u8_vocab_size = self.u8_tokenizer.vocab_size + len(self.u8_tokenizer.special_tokens)

    def chars_encode(self, text, add_special_tokens=False):
        return self.u8_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def chars_decode(self, tokens):
        return self.u8_tokenizer.decode(tokens)

    def gpt_encode(self, text, add_special_tokens=False):
        raw = self.gpt4o_tokenizer.encode(text)
        encoded = []
        if add_special_tokens:
            encoded.append(self.u8_tokenizer.bos_token_id)
        for r in raw:
            encoded.append(r + self.u8_vocab_size)
        if add_special_tokens:
            encoded.append(self.u8_tokenizer.eos_token_id)
        return encoded

    def gpt_decode(self, tokens):
        filtered_tokens = []
        for token in tokens:
            if token in self.u8_tokenizer.special_tokens.values():
                continue
            filtered_tokens.append(token - self.u8_vocab_size)
        return self.gpt4o_tokenizer.decode(filtered_tokens)


if __name__ == "__main__":
    mixed_tokenizer = MixedTokenizer()
    a = mixed_tokenizer.chars_encode("Hello world")
    print(a)
    b = mixed_tokenizer.gpt_encode("Hello world")
    print(b)
    c = mixed_tokenizer.chars_decode(a)
    print(c)
    d = mixed_tokenizer.gpt_decode(b)
    print(d)
