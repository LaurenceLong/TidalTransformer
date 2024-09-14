import tiktoken

from utf8_tokenizer import UTF8Tokenizer


class MixedTokenizer:
    def __init__(self):
        self.utf8_tokenizer = UTF8Tokenizer()
        self.gpt4o_tokenizer = tiktoken.get_encoding("o200k_base")  # GPT4o
        self.u8_vocab_size = self.utf8_tokenizer.vocab_size + len(self.utf8_tokenizer.special_tokens)
        self.vocab_size = self.u8_vocab_size + self.gpt4o_tokenizer.n_vocab
        self.eos_token_id = self.utf8_tokenizer.eos_token_id
        self.bos_token_id = self.utf8_tokenizer.bos_token_id
        self.pad_token_id = self.utf8_tokenizer.pad_token_id
        self.unk_token_id = self.utf8_tokenizer.unk_token_id
        self.bob_token_id = self.utf8_tokenizer.bob_token_id
        self.eob_token_id = self.utf8_tokenizer.eob_token_id

    def u8_encode(self, text, add_special_tokens=False):
        return self.utf8_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def u8_decode(self, tokens):
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        return self.utf8_tokenizer.decode(tokens)

    def encode(self, text, add_special_tokens=False):
        raw = self.gpt4o_tokenizer.encode(text)
        encoded = []
        if add_special_tokens:
            encoded.append(self.bos_token_id)
        for r in raw:
            encoded.append(r + self.u8_vocab_size)
        if add_special_tokens:
            encoded.append(self.eos_token_id)
        return encoded

    def decode(self, tokens):
        filtered_tokens = []
        for token in tokens:
            if token in self.utf8_tokenizer.special_tokens.values():
                continue
            filtered_tokens.append(token - self.u8_vocab_size)
        return self.gpt4o_tokenizer.decode(filtered_tokens)


if __name__ == "__main__":
    mixed_tokenizer = MixedTokenizer()
    a = mixed_tokenizer.u8_encode("Hello world")
    print(a)
    b = mixed_tokenizer.encode("Hello world")
    print(b)
    c = mixed_tokenizer.u8_decode(a)
    print(c)
    d = mixed_tokenizer.decode(b)
    print(d)
