"""Matrix visualization components for Streamlit."""

from typing import List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from utils.formatting import format_matrix_df


def render_matrix_card(
    matrix: np.ndarray,
    title: str,
    algebraic_label: Optional[str] = None,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    precision: int = 3,
    note: Optional[str] = None,
) -> None:
    """Render a styled card displaying a matrix, shape badge, and algebraic role."""
    shape_str = f"({matrix.shape[0]} × {matrix.shape[1]})" if matrix.ndim == 2 else f"{matrix.shape}"

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.markdown(f"**{title}** `{shape_str}`")
    if algebraic_label:
        st.caption(f"🏷️ *{algebraic_label}*")

    df = format_matrix_df(matrix, row_labels=row_labels, col_labels=col_labels, precision=precision)
    st.dataframe(df, use_container_width=True)

    if note:
        st.caption(f"💡 {note}")


def render_matrix_multiplication(
    mat_a: np.ndarray,
    label_a: str,
    mat_b: np.ndarray,
    label_b: str,
    mat_res: np.ndarray,
    label_res: str,
    op_symbol: str = "×",
    algebraic_label: Optional[str] = None,
    rows_a: Optional[List[str]] = None,
    cols_a: Optional[List[str]] = None,
    rows_b: Optional[List[str]] = None,
    cols_b: Optional[List[str]] = None,
    rows_res: Optional[List[str]] = None,
    cols_res: Optional[List[str]] = None,
    precision: int = 3,
) -> None:
    """Render side-by-side or stacked matrix operation: MatA [op] MatB = MatResult."""
    if algebraic_label:
        st.info(f"📐 **Aljabar Operasi**: {algebraic_label}")

    col1, col_op1, col2, col_op2, col3 = st.columns([4, 1, 4, 1, 4])

    with col1:
        st.markdown(f"**{label_a}** `({mat_a.shape[0]} × {mat_a.shape[1]})`")
        df_a = format_matrix_df(mat_a, row_labels=rows_a, col_labels=cols_a, precision=precision)
        st.dataframe(df_a, use_container_width=True)

    with col_op1:
        st.markdown("<br><br><h2 style='text-align: center; color: gray;'>" + op_symbol + "</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"**{label_b}** `({mat_b.shape[0]} × {mat_b.shape[1]})`")
        df_b = format_matrix_df(mat_b, row_labels=rows_b, col_labels=cols_b, precision=precision)
        st.dataframe(df_b, use_container_width=True)

    with col_op2:
        st.markdown("<br><br><h2 style='text-align: center; color: gray;'>=</h2>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"**{label_res}** `({mat_res.shape[0]} × {mat_res.shape[1]})`")
        df_res = format_matrix_df(mat_res, row_labels=rows_res, col_labels=cols_res, precision=precision)
        st.dataframe(df_res, use_container_width=True)
