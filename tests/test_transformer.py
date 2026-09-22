"""Automated test suite for Transformer numerical engine and mathematical properties."""

import unittest
import numpy as np

from transformer.tokenizer import SimpleTokenizer
from transformer.embedding import EmbeddingLayer
from transformer.positional_encoding import PositionalEncoding
from transformer.attention import MultiHeadAttention, softmax
from transformer.layer_norm import LayerNorm
from transformer.ffn import FeedForwardNetwork, gelu
from transformer.transformer_block import TransformerBlock


class TestTransformerEngine(unittest.TestCase):
    """Test mathematical accuracy, shape invariants, and reproducibility."""

    def setUp(self) -> None:
        self.d_model = 8
        self.num_heads = 2
        self.d_head = 4
        self.d_ff = 16
        self.epsilon = 1e-5
        self.seed = 42
        self.text = "Saya makan nasi goreng"

    def test_tokenizer(self) -> None:
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(self.text)
        self.assertEqual(tokens, ["Saya", "makan", "nasi", "goreng"])
        token_ids, vocab = tokenizer.encode(tokens)
        self.assertEqual(token_ids, [1, 2, 3, 4])
        self.assertEqual(tokenizer.decode(token_ids), tokens)

    def test_embedding_and_pe_shapes(self) -> None:
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(self.text)
        token_ids, vocab = tokenizer.encode(tokens)

        embed_layer = EmbeddingLayer(vocab_size=len(vocab), d_model=self.d_model, seed=self.seed)
        x_embed, E = embed_layer.forward(token_ids)
        self.assertEqual(x_embed.shape, (4, self.d_model))
        self.assertEqual(E.shape, (len(vocab), self.d_model))

        pe_layer = PositionalEncoding(d_model=self.d_model)
        X, pe = pe_layer.forward(x_embed)
        self.assertEqual(pe.shape, (4, self.d_model))
        self.assertEqual(X.shape, (4, self.d_model))
        np.testing.assert_allclose(X, x_embed + pe)

    def test_attention_shapes_and_softmax(self) -> None:
        mha = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_head=self.d_head,
            seed=self.seed,
        )
        rng = np.random.default_rng(self.seed)
        X = rng.normal(0, 1, size=(4, self.d_model))

        output, trace = mha.forward(X)
        self.assertEqual(output.shape, (4, self.d_model))
        self.assertEqual(trace.Z_concat.shape, (4, self.d_model))
        self.assertEqual(len(trace.heads), self.num_heads)

        for head in trace.heads:
            self.assertEqual(head.Q.shape, (4, self.d_head))
            self.assertEqual(head.K.shape, (4, self.d_head))
            self.assertEqual(head.V.shape, (4, self.d_head))
            self.assertEqual(head.scores.shape, (4, 4))
            self.assertEqual(head.attention_weights.shape, (4, 4))
            # Test softmax row sum == 1.0 (Section 32)
            row_sums = head.attention_weights.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(4), rtol=1e-5, atol=1e-6)

    def test_layernorm_properties(self) -> None:
        ln = LayerNorm(d_model=self.d_model, epsilon=self.epsilon)
        rng = np.random.default_rng(self.seed)
        x = rng.normal(5.0, 2.0, size=(4, self.d_model))

        y, trace = ln.forward(x)
        self.assertEqual(y.shape, (4, self.d_model))
        # With gamma=1 and beta=0, output mean per token must be ~0 and std/var ~1
        token_means = np.mean(y, axis=-1)
        token_vars = np.var(y, axis=-1)
        np.testing.assert_allclose(token_means, np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(token_vars, np.ones(4), rtol=1e-3, atol=1e-3)

    def test_ffn_shapes_and_gelu(self) -> None:
        ffn = FeedForwardNetwork(d_model=self.d_model, d_ff=self.d_ff, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        Y = rng.normal(0, 1, size=(4, self.d_model))

        output, trace = ffn.forward(Y)
        self.assertEqual(trace.H_pre.shape, (4, self.d_ff))
        self.assertEqual(trace.H.shape, (4, self.d_ff))
        self.assertEqual(output.shape, (4, self.d_model))

        # Check GELU property GELU(0) == 0
        zero_in = np.array([0.0])
        np.testing.assert_allclose(gelu(zero_in), np.zeros(1), atol=1e-6)

    def test_full_pipeline_trace(self) -> None:
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_head=self.d_head,
            d_ff=self.d_ff,
            seed=self.seed,
        )
        trace = block.forward(self.text)

        self.assertEqual(trace.tokens, ["Saya", "makan", "nasi", "goreng"])
        self.assertEqual(trace.token_ids, [1, 2, 3, 4])
        self.assertEqual(trace.X.shape, (4, 8))
        self.assertEqual(trace.R1.shape, (4, 8))
        self.assertEqual(trace.Y1.shape, (4, 8))
        self.assertEqual(trace.R2.shape, (4, 8))
        self.assertEqual(trace.Y2.shape, (4, 8))

        # Check required matrices are present in matrices_dict (Section 25)
        required_matrices = [
            "E", "PE", "X", "WQ1", "WK1", "WV1", "Q1", "K1", "V1", "Scores1", "A1", "Z1",
            "WQ2", "WK2", "WV2", "Q2", "K2", "V2", "Scores2", "A2", "Z2",
            "Concat", "WO", "MHA", "R1", "Y1", "W1", "H_pre", "H", "W2", "FFN", "R2", "Y2"
        ]
        for name in required_matrices:
            self.assertIn(name, trace.matrices_dict, f"Matrix {name} missing from trace")


if __name__ == "__main__":
    unittest.main()
