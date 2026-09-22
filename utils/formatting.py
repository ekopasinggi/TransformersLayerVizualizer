"""Formatting helpers for matrices, numbers, and educational LaTeX displays."""

from typing import List, Optional
import numpy as np
import pandas as pd


def format_num(val: float, precision: int = 3) -> str:
    """Format a single numeric value with given decimal precision."""
    if np.isnan(val):
        return "NaN"
    if precision is None or precision < 0:
        return f"{val:.6g}"
    return f"{val:.{precision}f}"


def format_matrix_df(
    matrix: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    precision: int = 3,
) -> pd.DataFrame:
    """Convert a 2D numpy array into a beautifully labelled pandas DataFrame."""
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    rows, cols = arr.shape

    if row_labels is None or len(row_labels) != rows:
        row_labels = [f"r{i}" for i in range(rows)]

    if col_labels is None or len(col_labels) != cols:
        col_labels = [f"c{j}" for j in range(cols)]

    df = pd.DataFrame(arr, index=row_labels, columns=col_labels)
    # Format floating point numbers to given precision
    if precision is not None:
        return df.map(lambda x: format_num(x, precision))
    return df


def latex_matrix(matrix: np.ndarray, max_rows: int = 6, max_cols: int = 8, precision: int = 3) -> str:
    """Generate LaTeX bmatrix string representation."""
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    rows, cols = arr.shape
    r_limit = min(rows, max_rows)
    c_limit = min(cols, max_cols)

    lines = []
    for i in range(r_limit):
        row_vals = [format_num(arr[i, j], precision) for j in range(c_limit)]
        if cols > c_limit:
            row_vals.append(r"\dots")
        lines.append(" & ".join(row_vals))

    if rows > r_limit:
        dots = [r"\vdots"] * c_limit
        if cols > c_limit:
            dots.append(r"\ddots")
        lines.append(" & ".join(dots))

    content = " \\\\\n".join(lines)
    return f"\\begin{{bmatrix}}\n{content}\n\\end{{bmatrix}}"


def render_html_safe(html_content: str) -> None:
    """Render HTML safely ensuring no leading whitespace triggers CommonMark code blocks."""
    import streamlit as st
    cleaned = "\n".join(line.strip() for line in html_content.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_edu_card(what: str, why: str, shape: str, aljabar: str) -> None:
    """Render an educational summary card with WHAT, WHY, SHAPE, and ALJABAR properties."""
    html = f"""
    <div class="edu-card">
        <b>WHAT:</b> {what}<br>
        <b>WHY:</b> {why}<br>
        <b>SHAPE:</b> <code>{shape}</code><br>
        <b>ALJABAR:</b> {aljabar}
    </div>
    """
    render_html_safe(html)

