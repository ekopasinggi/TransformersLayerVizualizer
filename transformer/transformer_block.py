"""Complete Post-LN Transformer Encoder Layer with comprehensive trace capture."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from .tokenizer import SimpleTokenizer
from .embedding import EmbeddingLayer
from .positional_encoding import PositionalEncoding
from .attention import MultiHeadAttention, AttentionTrace
from .layer_norm import LayerNorm, LayerNormTrace
from .ffn import FeedForwardNetwork, FFNTrace


@dataclass
class PipelineTrace:
    """Rich execution trace holding all intermediate tensors, shapes, and metadata."""

    raw_text: str
    tokens: List[str]
    token_ids: List[int]
    vocab: Dict[str, int]
    E: np.ndarray
    X_embed: np.ndarray
    PE: np.ndarray
    X: np.ndarray
    attention_trace: AttentionTrace
    R1: np.ndarray
    ln1_trace: LayerNormTrace
    Y1: np.ndarray
    ffn_trace: FFNTrace
    R2: np.ndarray
    ln2_trace: LayerNormTrace
    Y2: np.ndarray
    matrices_dict: Dict[str, Tuple[np.ndarray, Tuple[int, ...], str]]
    shapes_summary: List[Dict[str, str]]


class TransformerBlock:
    """Post-LN Single Layer Transformer Encoder."""

    def __init__(
        self,
        d_model: int = 8,
        num_heads: int = 2,
        d_head: int = 4,
        d_ff: int = 16,
        epsilon: float = 1e-5,
        seed: int = 42,
    ) -> None:
        if d_model != num_heads * d_head:
            raise ValueError(
                f"Invalid parameter configuration: d_model ({d_model}) != num_heads ({num_heads}) * d_head ({d_head})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.epsilon = epsilon
        self.seed = seed

        self.tokenizer = SimpleTokenizer()
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, d_head=d_head, seed=seed)
        self.ln1 = LayerNorm(d_model=d_model, epsilon=epsilon)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, seed=seed + 100)
        self.ln2 = LayerNorm(d_model=d_model, epsilon=epsilon)

    def forward(self, text: str) -> PipelineTrace:
        """Run complete forward pass and capture all computational states.

        Pipeline:
            1. Tokenization & Token IDs
            2. Embedding Lookup (E[ids])
            3. Positional Encoding (PE)
            4. Input Matrix X = X_embed + PE
            5. Multi-Head Attention: Q, K, V, Scaled Dot-Product, Softmax, Concat, W_O
            6. Residual 1: R1 = X + MHA(X)
            7. LayerNorm 1: Y1 = LN(R1)
            8. Position-wise FFN: H_pre = Y1 W1 + b1, H = GELU(H_pre), FFN = H W2 + b2
            9. Residual 2: R2 = Y1 + FFN(Y1)
            10. Final LayerNorm: Y2 = LN(R2)
        """
        # Step 1: Tokenization
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            tokens = ["<PAD>"]
        token_ids, vocab = self.tokenizer.encode(tokens, allow_dynamic_vocab=True)
        vocab_size = max(vocab.values()) + 1

        # Step 2: Embedding
        embedding_layer = EmbeddingLayer(vocab_size=vocab_size, d_model=self.d_model, seed=self.seed)
        X_embed, E = embedding_layer.forward(token_ids)

        # Step 3: Positional Encoding
        X, PE = self.pos_encoder.forward(X_embed)

        # Step 4: Multi-Head Attention
        MHA_output, att_trace = self.mha.forward(X)

        # Step 5: Residual Connection 1
        R1 = X + MHA_output

        # Step 6: LayerNorm 1
        Y1, ln1_trace = self.ln1.forward(R1)

        # Step 7: Feed-Forward Network
        FFN_output, ffn_trace = self.ffn.forward(Y1)

        # Step 8: Residual Connection 2
        R2 = Y1 + FFN_output

        # Step 9: Final LayerNorm 2
        Y2, ln2_trace = self.ln2.forward(R2)

        # Assemble comprehensive dictionary of all named matrices (Section 25)
        matrices_dict: Dict[str, Tuple[np.ndarray, Tuple[int, ...], str]] = {
            "E": (E, E.shape, "Embedding Table (vocab_size × d_model)"),
            "PE": (PE, PE.shape, "Sinusoidal Positional Encoding (seq_len × d_model)"),
            "X": (X, X.shape, "Input Representation X = X_embed + PE (seq_len × d_model)"),
        }

        # Add Head matrices
        for h in att_trace.heads:
            idx = h.head_idx
            matrices_dict[f"WQ{idx}"] = (h.W_Q, h.W_Q.shape, f"Query Projection Weights Head {idx} (d_model × d_head)")
            matrices_dict[f"WK{idx}"] = (h.W_K, h.W_K.shape, f"Key Projection Weights Head {idx} (d_model × d_head)")
            matrices_dict[f"WV{idx}"] = (h.W_V, h.W_V.shape, f"Value Projection Weights Head {idx} (d_model × d_head)")
            matrices_dict[f"Q{idx}"] = (h.Q, h.Q.shape, f"Query Matrix Head {idx} = X @ WQ{idx} (seq_len × d_head)")
            matrices_dict[f"K{idx}"] = (h.K, h.K.shape, f"Key Matrix Head {idx} = X @ WK{idx} (seq_len × d_head)")
            matrices_dict[f"V{idx}"] = (h.V, h.V.shape, f"Value Matrix Head {idx} = X @ WV{idx} (seq_len × d_head)")
            matrices_dict[f"Scores{idx}"] = (
                h.scores,
                h.scores.shape,
                f"Raw Attention Scores Head {idx} = (Q @ K.T) / sqrt(d_k)",
            )
            matrices_dict[f"A{idx}"] = (
                h.attention_weights,
                h.attention_weights.shape,
                f"Attention Weights Head {idx} = softmax(scores)",
            )
            matrices_dict[f"Z{idx}"] = (h.Z, h.Z.shape, f"Weighted Context Head {idx} = A @ V (seq_len × d_head)")

        matrices_dict["Concat"] = (
            att_trace.Z_concat,
            att_trace.Z_concat.shape,
            "Concatenated Head Outputs Concat(Z1, Z2, ...) (seq_len × d_model)",
        )
        matrices_dict["WO"] = (att_trace.W_O, att_trace.W_O.shape, "Output Projection Weights WO (d_model × d_model)")
        matrices_dict["MHA"] = (
            att_trace.MHA_output,
            att_trace.MHA_output.shape,
            "Multi-Head Attention Output MHA = Concat @ WO (seq_len × d_model)",
        )
        matrices_dict["R1"] = (R1, R1.shape, "Residual 1 = X + MHA(X) (seq_len × d_model)")
        matrices_dict["Y1"] = (Y1, Y1.shape, "Sublayer 1 Output = LayerNorm(R1) (seq_len × d_model)")
        matrices_dict["W1"] = (ffn_trace.W_1, ffn_trace.W_1.shape, "FFN Intermediate Weights W1 (d_model × d_ff)")
        matrices_dict["H_pre"] = (ffn_trace.H_pre, ffn_trace.H_pre.shape, "FFN Pre-Activation H_pre (seq_len × d_ff)")
        matrices_dict["H"] = (ffn_trace.H, ffn_trace.H.shape, "FFN Hidden State H = GELU(H_pre) (seq_len × d_ff)")
        matrices_dict["W2"] = (ffn_trace.W_2, ffn_trace.W_2.shape, "FFN Output Weights W2 (d_ff × d_model)")
        matrices_dict["FFN"] = (
            ffn_trace.FFN_output,
            ffn_trace.FFN_output.shape,
            "FFN Sublayer Output = H @ W2 + b2 (seq_len × d_model)",
        )
        matrices_dict["R2"] = (R2, R2.shape, "Residual 2 = Y1 + FFN(Y1) (seq_len × d_model)")
        matrices_dict["Y2"] = (
            Y2,
            Y2.shape,
            "Final Transformer Layer Output = LayerNorm(R2) (seq_len × d_model)",
        )

        # Assemble tabular shape summary for Shape Tracker (Section 26)
        shapes_summary: List[Dict[str, str]] = [
            {"Stage": "Embedding", "Tensor": "X_embed", "Shape": str(X_embed.shape)},
            {"Stage": "Positional Encoding", "Tensor": "PE", "Shape": str(PE.shape)},
            {"Stage": "Input Representation", "Tensor": "X", "Shape": str(X.shape)},
        ]
        for h in att_trace.heads:
            idx = h.head_idx
            shapes_summary.extend(
                [
                    {"Stage": f"Head {idx} Projections", "Tensor": f"Q{idx}, K{idx}, V{idx}", "Shape": str(h.Q.shape)},
                    {"Stage": f"Head {idx} Scores", "Tensor": f"Scores{idx}", "Shape": str(h.scores.shape)},
                    {"Stage": f"Head {idx} Attention", "Tensor": f"A{idx}", "Shape": str(h.attention_weights.shape)},
                    {"Stage": f"Head {idx} Output", "Tensor": f"Z{idx}", "Shape": str(h.Z.shape)},
                ]
            )
        shapes_summary.extend(
            [
                {"Stage": "Head Concatenation", "Tensor": "Z (Concat)", "Shape": str(att_trace.Z_concat.shape)},
                {"Stage": "MHA Output Projection", "Tensor": "MHA", "Shape": str(att_trace.MHA_output.shape)},
                {"Stage": "Residual 1", "Tensor": "R1 (X + MHA)", "Shape": str(R1.shape)},
                {"Stage": "LayerNorm 1", "Tensor": "Y1", "Shape": str(Y1.shape)},
                {"Stage": "FFN Expansion", "Tensor": "H_pre, H", "Shape": str(ffn_trace.H.shape)},
                {"Stage": "FFN Projection", "Tensor": "FFN", "Shape": str(ffn_trace.FFN_output.shape)},
                {"Stage": "Residual 2", "Tensor": "R2 (Y1 + FFN)", "Shape": str(R2.shape)},
                {"Stage": "LayerNorm 2 (Final)", "Tensor": "Y2", "Shape": str(Y2.shape)},
            ]
        )

        return PipelineTrace(
            raw_text=text,
            tokens=tokens,
            token_ids=token_ids,
            vocab=vocab,
            E=E,
            X_embed=X_embed,
            PE=PE,
            X=X,
            attention_trace=att_trace,
            R1=R1,
            ln1_trace=ln1_trace,
            Y1=Y1,
            ffn_trace=ffn_trace,
            R2=R2,
            ln2_trace=ln2_trace,
            Y2=Y2,
            matrices_dict=matrices_dict,
            shapes_summary=shapes_summary,
        )
