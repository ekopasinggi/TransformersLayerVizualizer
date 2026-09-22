"""Position-wise Feed-Forward Network with GELU activation."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class FFNTrace:
    """Trace of all intermediate activations and weights for Feed-Forward Network."""

    Y_input: np.ndarray  # (seq_len, d_model)
    W_1: np.ndarray  # (d_model, d_ff)
    b_1: np.ndarray  # (d_ff,)
    H_pre: np.ndarray  # (seq_len, d_ff)
    H: np.ndarray  # (seq_len, d_ff) after GELU
    W_2: np.ndarray  # (d_ff, d_model)
    b_2: np.ndarray  # (d_model,)
    FFN_output: np.ndarray  # (seq_len, d_model)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function using standard tanh approximation.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    inner = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


class FeedForwardNetwork:
    """Position-wise Feed-Forward Network operating independently on each token representation."""

    def __init__(self, d_model: int = 8, d_ff: int = 16, seed: int = 42) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.seed = seed
        self._init_weights()

    def _init_weights(self) -> None:
        """Deterministic initialization of W1, b1, W2, b2."""
        rng = np.random.default_rng(self.seed)
        std1 = 1.0 / np.sqrt(self.d_model)
        std2 = 1.0 / np.sqrt(self.d_ff)

        self.W_1 = rng.normal(0.0, std1, size=(self.d_model, self.d_ff))
        self.b_1 = np.zeros(self.d_ff, dtype=np.float64)

        self.W_2 = rng.normal(0.0, std2, size=(self.d_ff, self.d_model))
        self.b_2 = np.zeros(self.d_model, dtype=np.float64)

    def forward(self, Y: np.ndarray) -> Tuple[np.ndarray, FFNTrace]:
        """Compute position-wise FFN forward pass:

        H_pre = Y @ W1 + b1
        H = GELU(H_pre)
        FFN = H @ W2 + b2
        """
        # Linear layer 1: (seq_len, d_ff)
        H_pre = Y @ self.W_1 + self.b_1

        # Non-linear activation: (seq_len, d_ff)
        H = gelu(H_pre)

        # Linear layer 2: (seq_len, d_model)
        FFN_output = H @ self.W_2 + self.b_2

        trace = FFNTrace(
            Y_input=Y,
            W_1=self.W_1,
            b_1=self.b_1,
            H_pre=H_pre,
            H=H,
            W_2=self.W_2,
            b_2=self.b_2,
            FFN_output=FFN_output,
        )

        return FFN_output, trace
