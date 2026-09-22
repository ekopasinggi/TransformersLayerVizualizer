"""Sinusoidal Positional Encoding for Transformer encoder."""

from typing import Tuple
import numpy as np


class PositionalEncoding:
    """Computes standard Vaswani et al. (2017) sinusoidal positional encodings."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        self.d_model = d_model
        self.max_len = max_len

    def compute(self, seq_len: int) -> np.ndarray:
        """Calculate PE table of shape (seq_len, d_model).

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        """
        pe = np.zeros((seq_len, self.d_model), dtype=np.float64)
        position = np.arange(0, seq_len, dtype=np.float64)[:, np.newaxis]

        # Division term for even indices: 2i
        # 10000^(2i / d_model) = exp((2i / d_model) * ln(10000))
        div_indices = np.arange(0, self.d_model, 2, dtype=np.float64)
        div_term = np.exp(div_indices * (-np.log(10000.0) / self.d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        # Handle odd indices (ensure dimension matching if d_model is odd)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, x_embed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply positional encoding to token embeddings.

        Returns:
            X: x_embed + PE, shape (seq_len, d_model)
            PE: Positional encoding matrix, shape (seq_len, d_model)
        """
        seq_len = x_embed.shape[0]
        pe = self.compute(seq_len)
        x = x_embed + pe
        return x, pe
