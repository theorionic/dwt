"""GPT-2 BPE tokenizer wrapper using tiktoken."""

import tiktoken


class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab
        self.eos_token_id = self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self._enc.encode(t) for t in texts]


_tokenizer: BPETokenizer | None = None


def get_tokenizer() -> BPETokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BPETokenizer()
    return _tokenizer
