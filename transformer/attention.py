"""Multi-Head Self-Attention with transparent per-head projections and scaling."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class HeadTrace:
    """Trace of all intermediate matrices for a single attention head."""

    head_idx: int
    W_Q: np.ndarray  # (d_model, d_head)
    W_K: np.ndarray  # (d_model, d_head)
    W_V: np.ndarray  # (d_model, d_head)
    Q: np.ndarray  # (seq_len, d_head)
    K: np.ndarray  # (seq_len, d_head)
    V: np.ndarray  # (seq_len, d_head)
    scores: np.ndarray  # (seq_len, seq_len)
    attention_weights: np.ndarray  # (seq_len, seq_len)
    Z: np.ndarray  # (seq_len, d_head)


@dataclass
class AttentionTrace:
    """Complete trace of Multi-Head Attention computation."""

    heads: List[HeadTrace]
    Z_concat: np.ndarray  # (seq_len, d_model)
    W_O: np.ndarray  # (d_model, d_model)
    MHA_output: np.ndarray  # (seq_len, d_model)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """Pure NumPy Multi-Head Attention adhering strictly to pedagogical specification."""

    def __init__(self, d_model: int = 8, num_heads: int = 2, d_head: int = 4, seed: int = 42) -> None:
        if d_model != num_heads * d_head:
            raise ValueError(
                f"Dimension mismatch: d_model ({d_model}) must equal num_heads ({num_heads}) * d_head ({d_head})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.seed = seed
        self.scale = 1.0 / np.sqrt(self.d_head)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize distinct deterministic weight matrices per head and output projection."""
        rng = np.random.default_rng(self.seed)
        std_head = 1.0 / np.sqrt(self.d_model)

        self.W_Q_heads: List[np.ndarray] = [
            rng.normal(0.0, std_head, size=(self.d_model, self.d_head)) for _ in range(self.num_heads)
        ]
        self.W_K_heads: List[np.ndarray] = [
            rng.normal(0.0, std_head, size=(self.d_model, self.d_head)) for _ in range(self.num_heads)
        ]
        self.W_V_heads: List[np.ndarray] = [
            rng.normal(0.0, std_head, size=(self.d_model, self.d_head)) for _ in range(self.num_heads)
        ]

        std_out = 1.0 / np.sqrt(self.d_model)
        self.W_O = rng.normal(0.0, std_out, size=(self.d_model, self.d_model))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, AttentionTrace]:
        """Execute forward pass for Multi-Head Self-Attention.

        Args:
            X: Input matrix of shape (seq_len, d_model)

        Returns:
            MHA_output: Output after projection (seq_len, d_model)
            trace: AttentionTrace containing all intermediate tensors for inspection
        """
        seq_len, d_m = X.shape
        if d_m != self.d_model:
            raise ValueError(f"Input feature dimension {d_m} != d_model {self.d_model}")

        head_traces: List[HeadTrace] = []
        head_outputs: List[np.ndarray] = []

        for i in range(self.num_heads):
            # Principle: Full X is projected into each head's space
            W_Q = self.W_Q_heads[i]
            W_K = self.W_K_heads[i]
            W_V = self.W_V_heads[i]

            Q = X @ W_Q  # (seq_len, d_head)
            K = X @ W_K  # (seq_len, d_head)
            V = X @ W_V  # (seq_len, d_head)

            # Scaled Dot-Product Attention
            scores = (Q @ K.T) * self.scale  # (seq_len, seq_len)
            A = softmax(scores, axis=-1)  # (seq_len, seq_len)
            Z = A @ V  # (seq_len, d_head)

            head_outputs.append(Z)
            head_traces.append(
                HeadTrace(
                    head_idx=i + 1,
                    W_Q=W_Q,
                    W_K=W_K,
                    W_V=W_V,
                    Q=Q,
                    K=K,
                    V=V,
                    scores=scores,
                    attention_weights=A,
                    Z=Z,
                )
            )

        # Concatenate head outputs along feature axis
        Z_concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)

        # Output projection
        MHA_output = Z_concat @ self.W_O  # (seq_len, d_model)

        trace = AttentionTrace(
            heads=head_traces,
            Z_concat=Z_concat,
            W_O=self.W_O,
            MHA_output=MHA_output,
        )

        return MHA_output, trace
