"""Pipeline numerical validation tests."""

from dataclasses import dataclass
from typing import List
import numpy as np

from transformer.transformer_block import PipelineTrace


@dataclass
class ValidationItem:
    """Individual assertion check result."""

    category: str
    check_name: str
    passed: bool
    details: str


def validate_pipeline(trace: PipelineTrace, d_model: int, num_heads: int, d_head: int, d_ff: int) -> List[ValidationItem]:
    """Execute all Section 32 mathematical and dimensional validations."""
    results: List[ValidationItem] = []

    # 1. MHA Dimension rule
    mha_dim_valid = (d_model == num_heads * d_head)
    results.append(
        ValidationItem(
            category="Multi-Head Attention",
            check_name="d_model == num_heads * d_head",
            passed=mha_dim_valid,
            details=f"{d_model} == {num_heads} * {d_head} ({'Match' if mha_dim_valid else 'Mismatch'})",
        )
    )

    # 2. Softmax row-sum == 1.0 for each head
    for head in trace.attention_trace.heads:
        row_sums = head.attention_weights.sum(axis=1)
        close_to_one = bool(np.allclose(row_sums, 1.0, atol=1e-5))
        max_diff = float(np.max(np.abs(row_sums - 1.0)))
        results.append(
            ValidationItem(
                category=f"Attention Head {head.head_idx}",
                check_name="Softmax Row Sums == 1.0",
                passed=close_to_one,
                details=f"All rows sum to ~1.0 (max deviation: {max_diff:.2e})",
            )
        )

    # 3. Residual 1 shape match
    r1_shape_match = (trace.X.shape == trace.attention_trace.MHA_output.shape == trace.R1.shape)
    results.append(
        ValidationItem(
            category="Residual Connection 1",
            check_name="Shape compatibility: X + MHA(X)",
            passed=r1_shape_match,
            details=f"X: {trace.X.shape}, MHA: {trace.attention_trace.MHA_output.shape} -> R1: {trace.R1.shape}",
        )
    )

    # 4. LayerNorm 1 properties (mean ~ 0, var ~ 1)
    ln1_means = np.mean(trace.Y1, axis=-1)
    ln1_vars = np.var(trace.Y1, axis=-1)
    ln1_mean_zero = bool(np.allclose(ln1_means, 0.0, atol=1e-4))
    ln1_var_one = bool(np.allclose(ln1_vars, 1.0, atol=1e-2))
    results.append(
        ValidationItem(
            category="LayerNorm 1",
            check_name="Mean ~ 0.0 & Variance ~ 1.0 across d_model",
            passed=(ln1_mean_zero and ln1_var_one),
            details=f"Token means in [{np.min(ln1_means):.4f}, {np.max(ln1_means):.4f}], vars in [{np.min(ln1_vars):.4f}, {np.max(ln1_vars):.4f}]",
        )
    )

    # 5. FFN Dimensions
    ffn_dim_valid = (
        trace.ffn_trace.W_1.shape == (d_model, d_ff)
        and trace.ffn_trace.H.shape == (len(trace.tokens), d_ff)
        and trace.ffn_trace.W_2.shape == (d_ff, d_model)
        and trace.ffn_trace.FFN_output.shape == (len(trace.tokens), d_model)
    )
    results.append(
        ValidationItem(
            category="Feed-Forward Network",
            check_name="Dimension projection: d_model -> d_ff -> d_model",
            passed=ffn_dim_valid,
            details=f"Y1 ({d_model}) -> H ({d_ff}) -> FFN ({d_model})",
        )
    )

    # 6. Residual 2 shape match
    r2_shape_match = (trace.Y1.shape == trace.ffn_trace.FFN_output.shape == trace.R2.shape)
    results.append(
        ValidationItem(
            category="Residual Connection 2",
            check_name="Shape compatibility: Y1 + FFN(Y1)",
            passed=r2_shape_match,
            details=f"Y1: {trace.Y1.shape}, FFN: {trace.ffn_trace.FFN_output.shape} -> R2: {trace.R2.shape}",
        )
    )

    # 7. LayerNorm 2 properties
    ln2_means = np.mean(trace.Y2, axis=-1)
    ln2_vars = np.var(trace.Y2, axis=-1)
    ln2_mean_zero = bool(np.allclose(ln2_means, 0.0, atol=1e-4))
    ln2_var_one = bool(np.allclose(ln2_vars, 1.0, atol=1e-2))
    results.append(
        ValidationItem(
            category="LayerNorm 2 (Final)",
            check_name="Mean ~ 0.0 & Variance ~ 1.0 across d_model",
            passed=(ln2_mean_zero and ln2_var_one),
            details=f"Token means in [{np.min(ln2_means):.4f}, {np.max(ln2_means):.4f}], vars in [{np.min(ln2_vars):.4f}, {np.max(ln2_vars):.4f}]",
        )
    )

    return results
