"""Interactive Plotly Attention Heatmap component."""

from typing import List
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    head_title: str = "Attention Head 1",
    precision: int = 3,
) -> None:
    """Render interactive attention heatmap with rich tooltips and row-sum validation."""
    n_tokens = len(tokens)

    # Prepare custom hovertext
    hover_text = []
    text_display = []
    for i in range(n_tokens):
        row_hover = []
        row_display = []
        for j in range(n_tokens):
            val = attention_matrix[i, j]
            row_hover.append(
                f"<b>Query token:</b> {tokens[i]}<br>"
                f"<b>Key token:</b> {tokens[j]}<br>"
                f"<b>Attention weight:</b> {val:.{precision}f}"
            )
            row_display.append(f"{val:.{precision}f}")
        hover_text.append(row_hover)
        text_display.append(row_display)

    fig = go.Figure(
        data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            text=text_display,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            hoverinfo="text",
            hovertext=hover_text,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Weight"),
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{head_title} — Attention Weights Matrix</b>", font=dict(size=16)),
        xaxis=dict(title="<b>Key Tokens (K)</b>", side="bottom"),
        yaxis=dict(title="<b>Query Tokens (Q)</b>", autorange="reversed"),
        width=550,
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Row sums validation table (Section 12: row_sum = A.sum(axis=1))
    row_sums = attention_matrix.sum(axis=1)
    sum_cols = st.columns(n_tokens)
    for idx, (tok, r_sum) in enumerate(zip(tokens, row_sums)):
        with sum_cols[idx]:
            st.metric(
                label=f"Σ Row '{tok}'",
                value=f"{r_sum:.{precision}f}",
                delta="1.000 (Softmax OK)" if np.isclose(r_sum, 1.0, atol=1e-4) else "Deviation",
            )
