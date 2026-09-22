"""Embedding lookup layer for Transformer encoder."""

from typing import List, Tuple
import numpy as np


class EmbeddingLayer:
    """Computes token embeddings from a deterministic vocabulary embedding table."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seed = seed
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> np.ndarray:
        """Initialize deterministic embedding table E ~ N(0, 1 / sqrt(d_model))."""
        rng = np.random.default_rng(self.seed)
        # Using normalized normal distribution scaled for stable magnitude
        std = 1.0 / np.sqrt(self.d_model)
        weights = rng.normal(loc=0.0, scale=std, size=(self.vocab_size, self.d_model))
        # Zero out PAD embedding if index 0
        weights[0] = 0.0
        return weights

    def forward(self, token_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Lookup embeddings for token_ids.

        Returns:
            X_embed: Array of shape (len(token_ids), d_model)
            E: Full embedding matrix (vocab_size, d_model)
        """
        arr_ids = np.array(token_ids, dtype=np.int64)
        x_embed = self.weights[arr_ids]
        return x_embed, self.weights
