"""Calculation inspection views providing cell-level and token-level algebraic breakdowns."""

from typing import List, Optional
import numpy as np
import streamlit as st

from utils.formatting import format_num


def render_matmul_element_detail(
    mat_a: np.ndarray,
    name_a: str,
    mat_b: np.ndarray,
    name_b: str,
    mat_c: np.ndarray,
    name_c: str,
    row_idx: int,
    col_idx: int,
    row_label: Optional[str] = None,
    col_label: Optional[str] = None,
    precision: int = 3,
) -> None:
    """Show detailed linear combination breakdown for a single cell C[i, j] = A[i, :] @ B[:, j]."""
    k_dim = mat_a.shape[1]
    terms_formula = []
    terms_numeric = []
    product_values = []

    for k in range(k_dim):
        a_val = mat_a[row_idx, k]
        b_val = mat_b[k, col_idx]
        prod = a_val * b_val
        product_values.append(prod)

        terms_formula.append(f"{name_a}[{row_idx},{k}] · {name_b}[{k},{col_idx}]")
        terms_numeric.append(f"({format_num(a_val, precision)} × {format_num(b_val, precision)})")

    sum_val = float(np.sum(product_values))
    actual_c = float(mat_c[row_idx, col_idx])

    label_str = f" ('{row_label}', '{col_label}')" if (row_label and col_label) else ""
    st.markdown(f"#### 🔍 Perhitungan Detail Sel `{name_c}[{row_idx}, {col_idx}]`{label_str}")

    st.markdown("**1. Formula Aljabar (Dot Product):**")
    st.latex(f"{name_c}[{row_idx},{col_idx}] = \\sum_{{k=0}}^{{{k_dim - 1}}} {name_a}[{row_idx}, k] \\cdot {name_b}[k, {col_idx}]")

    st.markdown("**2. Substitusi Simbolik:**")
    st.code(" + \n".join(terms_formula), language="text")

    st.markdown("**3. Substitusi Nilai Numerik:**")
    numeric_str = " + \n".join(terms_numeric)
    products_str = " + ".join([format_num(p, precision) for p in product_values])
    st.code(f"= {numeric_str}\n\n= {products_str}\n\n= {format_num(sum_val, precision)}", language="text")

    st.success(f"✅ Hasil Komputasi: `{format_num(actual_c, precision)}`")


def render_attention_score_detail(
    Q: np.ndarray,
    K: np.ndarray,
    tokens: List[str],
    q_idx: int,
    k_idx: int,
    d_head: int,
    precision: int = 3,
) -> None:
    """Show detailed scaled dot-product calculation for Query[q_idx] and Key[k_idx]."""
    q_tok = tokens[q_idx]
    k_tok = tokens[k_idx]
    q_vec = Q[q_idx]
    k_vec = K[k_idx]

    scale = np.sqrt(d_head)
    dot_products = q_vec * k_vec
    raw_dot = float(np.sum(dot_products))
    scaled_score = raw_dot / scale

    st.markdown(f"#### 🔍 Perhitungan Scaled Dot-Product: Query `'{q_tok}'` · Key `'{k_tok}'`")

    st.markdown("**1. Vektor Representasi:**")
    col_q, col_k = st.columns(2)
    with col_q:
        q_str = ", ".join([format_num(v, precision) for v in q_vec])
        st.code(f"q_{{{q_tok}}} = [{q_str}]", language="text")
    with col_k:
        k_str = ", ".join([format_num(v, precision) for v in k_vec])
        st.code(f"k_{{{k_tok}}} = [{k_str}]", language="text")

    st.markdown("**2. Dot Product (Alignment Score):**")
    expansion = " + ".join([f"({format_num(q, precision)} × {format_num(k, precision)})" for q, k in zip(q_vec, k_vec)])
    prod_terms = " + ".join([format_num(p, precision) for p in dot_products])
    st.latex(r"\text{DotProduct} = q \cdot k = \sum_{i=1}^{d_{head}} q_i \cdot k_i")
    st.code(f"= {expansion}\n= {prod_terms}\n= {format_num(raw_dot, precision)}", language="text")

    st.markdown("**3. Penskalaan dengan $\\sqrt{d_{head}}$:**")
    st.latex(r"\text{Score} = \frac{q \cdot k}{\sqrt{d_{head}}}")
    st.code(f"= {format_num(raw_dot, precision)} / sqrt({d_head})\n= {format_num(raw_dot, precision)} / {format_num(scale, precision)}\n= {format_num(scaled_score, precision)}", language="text")
    st.info(f"Skor sebelum softmax = **{format_num(scaled_score, precision)}**")


def render_layernorm_token_detail(
    x_input: np.ndarray,
    token_label: str,
    token_idx: int,
    epsilon: float = 1e-5,
    gamma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    precision: int = 3,
) -> None:
    """Show step-by-step Layer Normalization arithmetic for a single token."""
    x = x_input[token_idx]
    d = len(x)
    mean = float(np.mean(x))
    variance = float(np.mean((x - mean) ** 2))
    std_eps = float(np.sqrt(variance + epsilon))
    x_hat = (x - mean) / std_eps

    if gamma is None:
        gamma = np.ones(d)
    if beta is None:
        beta = np.zeros(d)
    y = gamma * x_hat + beta

    st.markdown(f"#### 🔍 Perhitungan Rinci LayerNorm untuk Token: `'{token_label}'` (Index {token_idx})")

    st.markdown("**1. Input Vector $x$:**")
    st.code(f"x = [{', '.join([format_num(v, precision) for v in x])}]", language="text")

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Mean (μ)", format_num(mean, precision))
        st.latex(r"\mu = \frac{1}{d} \sum_{i=1}^d x_i")
    with col_stat2:
        st.metric("Variance (σ²)", format_num(variance, precision))
        st.latex(r"\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2")
    with col_stat3:
        st.metric("Std Dev √(σ² + ε)", format_num(std_eps, precision))
        st.latex(r"\sqrt{\sigma^2 + \epsilon}")

    st.markdown("**2. Normalized Vector $\\hat{x} = (x - \\mu) / \\sqrt{\\sigma^2 + \\epsilon}$:**")
    st.code(f"x_hat = [{', '.join([format_num(v, precision) for v in x_hat])}]", language="text")

    st.markdown("**3. Scale & Shift ($y = \\gamma \\odot \\hat{x} + \\beta$):**")
    st.code(f"gamma = [{', '.join([format_num(g, precision) for g in gamma])}]\nbeta  = [{', '.join([format_num(b, precision) for b in beta])}]\ny     = [{', '.join([format_num(v, precision) for v in y])}]", language="text")
    st.success(f"Rata-rata output token = **{format_num(float(np.mean(y)), 4)}** (mendekati 0), Varian = **{format_num(float(np.var(y)), 4)}** (mendekati 1)")


def render_gelu_element_detail(x_val: float, precision: int = 3) -> None:
    """Show detailed evaluation of GELU activation approximation for a scalar value."""
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    x3 = x_val**3
    inner = sqrt_2_pi * (x_val + 0.044715 * x3)
    tanh_val = np.tanh(inner)
    gelu_val = 0.5 * x_val * (1.0 + tanh_val)

    st.markdown("#### 🔍 Detail Perhitungan Aktivasi GELU")
    st.latex(r"\text{GELU}(x) \approx 0.5 \cdot x \cdot \left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right]")

    st.code(
        f"Input x               = {format_num(x_val, precision)}\n"
        f"x³                    = {format_num(x3, precision)}\n"
        f"x + 0.044715 * x³     = {format_num(x_val + 0.044715 * x3, precision)}\n"
        f"Inner (sqrt(2/pi)*..) = {format_num(inner, precision)}\n"
        f"tanh(Inner)           = {format_num(tanh_val, precision)}\n"
        f"1 + tanh(...)         = {format_num(1.0 + tanh_val, precision)}\n"
        f"Output GELU(x)        = {format_num(gelu_val, precision)}",
        language="text",
    )
