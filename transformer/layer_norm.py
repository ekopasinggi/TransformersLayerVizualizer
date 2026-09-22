"""Layer Normalization performed per-token across feature dimension."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class LayerNormTrace:
    """Trace of all intermediate statistics and vectors for Layer Normalization."""

    x_input: np.ndarray  # (seq_len, d_model)
    mean: np.ndarray  # (seq_len, 1)
    variance: np.ndarray  # (seq_len, 1)
    std_eps: np.ndarray  # (seq_len, 1) sqrt(variance + epsilon)
    x_hat: np.ndarray  # (seq_len, d_model)
    gamma: np.ndarray  # (d_model,)
    beta: np.ndarray  # (d_model,)
    output: np.ndarray  # (seq_len, d_model)
    epsilon: float


class LayerNorm:
    """Per-token Layer Normalization across the d_model dimension."""

    def __init__(self, d_model: int, epsilon: float = 1e-5) -> None:
        self.d_model = d_model
        self.epsilon = epsilon
        # Default learnable parameters
        self.gamma = np.ones(d_model, dtype=np.float64)
        self.beta = np.zeros(d_model, dtype=np.float64)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, LayerNormTrace]:
        """Normalize each token vector independently across feature dimension d_model.

        Args:
            x: Input tensor of shape (seq_len, d_model)

        Returns:
            output: Normalized and scaled tensor (seq_len, d_model)
            trace: LayerNormTrace containing intermediate means, variances, x_hat, etc.
        """
        # Mean per token: (seq_len, 1)
        mean = np.mean(x, axis=-1, keepdims=True)

        # Variance per token: (seq_len, 1)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)

        # Standard deviation with epsilon
        std_eps = np.sqrt(variance + self.epsilon)

        # Normalized values
        x_hat = (x - mean) / std_eps

        # Scale and shift
        output = self.gamma * x_hat + self.beta

        trace = LayerNormTrace(
            x_input=x,
            mean=mean,
            variance=variance,
            std_eps=std_eps,
            x_hat=x_hat,
            gamma=self.gamma,
            beta=self.beta,
            output=output,
            epsilon=self.epsilon,
        )

        return output, trace
