"""Simple word-level tokenizer for educational visualization."""

from typing import Dict, List, Tuple


class SimpleTokenizer:
    """A clean, educational word-level tokenizer with base and extensible vocabulary."""

    DEFAULT_VOCAB: Dict[str, int] = {
        "<PAD>": 0,
        "Saya": 1,
        "makan": 2,
        "nasi": 3,
        "goreng": 4,
    }

    def __init__(self, custom_vocab: Dict[str, int] | None = None) -> None:
        self.vocab: Dict[str, int] = dict(self.DEFAULT_VOCAB)
        if custom_vocab:
            self.vocab.update(custom_vocab)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        """Split text into whitespace-delimited tokens while stripping excessive punctuation."""
        cleaned = text.strip()
        if not cleaned:
            return []
        return cleaned.split()

    def encode(self, tokens: List[str], allow_dynamic_vocab: bool = True) -> Tuple[List[int], Dict[str, int]]:
        """Convert tokens to IDs.

        If allow_dynamic_vocab is True, unseen tokens are appended deterministically to
        the vocabulary to allow custom input sentences.
        """
        token_ids: List[int] = []
        for token in tokens:
            if token not in self.vocab:
                if allow_dynamic_vocab:
                    next_id = max(self.vocab.values(), default=0) + 1
                    self.vocab[token] = next_id
                    self.id_to_token[next_id] = token
                    token_ids.append(next_id)
                else:
                    # Fallback to PAD if dynamic vocab disabled
                    token_ids.append(self.vocab["<PAD>"])
            else:
                token_ids.append(self.vocab[token])
        return token_ids, self.vocab

    def decode(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs back to tokens."""
        return [self.id_to_token.get(i, "<UNK>") for i in token_ids]
