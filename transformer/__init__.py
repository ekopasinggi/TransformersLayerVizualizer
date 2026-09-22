"""Transformer numerical engine package.

Contains modular components for a 1-layer Post-LN Transformer Encoder:
- Tokenizer & Embedding
- Sinusoidal Positional Encoding
- Multi-Head Self-Attention (pure NumPy, no splitting prior to projection)
- Layer Normalization (per-token over d_model)
- Feed-Forward Network with GELU
- TransformerBlock orchestrator with complete forward pass state recording
"""

from .tokenizer import SimpleTokenizer
from .embedding import EmbeddingLayer
from .positional_encoding import PositionalEncoding
from .attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .ffn import FeedForwardNetwork
from .transformer_block import TransformerBlock, PipelineTrace

__all__ = [
    "SimpleTokenizer",
    "EmbeddingLayer",
    "PositionalEncoding",
    "MultiHeadAttention",
    "LayerNorm",
    "FeedForwardNetwork",
    "TransformerBlock",
    "PipelineTrace",
]
