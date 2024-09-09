import time

import tiktoken

# encoding = tiktoken.get_encoding("cl100k_base")  # GPT4
encoding = tiktoken.get_encoding("o200k_base")  # GPT4o
print(encoding.max_token_value)

t0 = time.time()
for i in range(1000):
    res = encoding.encode("hello world")
print(time.time() - t0)
