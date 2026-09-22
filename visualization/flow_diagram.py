"""Pipeline flowchart visualizer with active step highlighting according to Section 24."""

from typing import List, Tuple
import streamlit as st

# 12 Pipeline steps with corresponding Post-LN architectural block names
STAGES: List[Tuple[int, str, str, str]] = [
    (1, "Input & Tokens", "Tokenisasi teks ke ID integer", "Input"),
    (2, "Embedding Matrix", "Lookup representasi vektor E[ids]", "Embedding"),
    (3, "Positional Encoding", "Injeksi urutan sinusoidal PE", "Positional Encoding"),
    (4, "Matriks Input X", "X = Embedding + PE", "X"),
    (5, "Proyeksi Q, K, V", "Linear transformation untuk tiap Head", "MHA"),
    (6, "Scaled Dot-Product", "Scores = (Q @ K.T) / sqrt(d_k)", "MHA"),
    (7, "Attention Weights", "Softmax normalisasi & heatmap", "MHA"),
    (8, "Head Concat", "Z = A @ V, Concat(Z1, Z2)", "MHA"),
    (9, "Output Projection", "MHA = Concat @ WO", "MHA"),
    (10, "Residual 1 + LayerNorm 1", "R1 = X + MHA lalu LayerNorm(R1)", "Residual & LayerNorm"),
    (11, "Feed-Forward Network", "H = GELU(Y1 W1 + b1), FFN = H W2 + b2", "FFN"),
    (12, "Residual 2 + LayerNorm 2", "R2 = Y1 + FFN lalu LayerNorm(R2)", "Residual & LayerNorm (Output)"),
]

# Section 24 specific architectural blocks
ARCH_BLOCKS: List[Tuple[str, List[int], str]] = [
    ("Input", [1], "Teks input 'Saya makan nasi goreng' & token IDs"),
    ("Embedding", [2], "Tabel embedding E (vocab_size × d_model)"),
    ("Positional Encoding", [3], "Sinusoidal encoding PE (seq_len × d_model)"),
    ("X", [4], "Matriks representasi awal X = X_embed + PE"),
    ("MHA", [5, 6, 7, 8, 9], "Multi-Head Attention (Q, K, V, Scaled Dot-Product, Softmax, Concat, WO)"),
    ("Residual", [10], "Residual Connection 1: R1 = X + MHA(X)"),
    ("LayerNorm", [10], "Layer Normalization 1: Y1 = LayerNorm(R1)"),
    ("FFN", [11], "Position-wise Feed-Forward Network dengan GELU"),
    ("Residual", [12], "Residual Connection 2: R2 = Y1 + FFN(Y1)"),
    ("LayerNorm", [12], "Layer Normalization 2: Y2 = LayerNorm(R2)"),
    ("Output", [12], "Output akhir Transformer Encoder Layer"),
]


def render_html_safe(html_content: str) -> None:
    """Render HTML safely ensuring no leading whitespace triggers CommonMark code blocks."""
    cleaned = "\n".join(line.strip() for line in html_content.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_pipeline_flow(current_step: int) -> None:
    """Render modern visual pipeline illustration with active step highlighting."""
    st.markdown("### 🗺️ Arsitektur Pipeline Post-LN Transformer Encoder")

    # Render horizontal interactive cards
    card_elements = []
    for step_num, title, desc, _ in STAGES:
        is_active = (step_num == current_step)
        if is_active:
            bg_style = "background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%); border: 2px solid #00e5ff; box-shadow: 0 0 12px rgba(0, 229, 255, 0.6);"
            badge_text = f"✨ LANGKAH {step_num} [AKTIF]"
            badge_color = "#00e5ff"
            title_color = "#ffffff"
            desc_color = "#e0f7fa"
        else:
            bg_style = "background: #1e1e2d; border: 1px solid #323248; box-shadow: none;"
            badge_text = f"Langkah {step_num}"
            badge_color = "#8f90a6"
            title_color = "#ffffff"
            desc_color = "#9fa0b5"

        card_html = f"""
        <div style="flex: 1 1 150px; min-width: 140px; max-width: 180px; {bg_style} border-radius: 8px; padding: 10px 12px; margin: 4px; box-sizing: border-box;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {badge_color}; text-transform: uppercase; letter-spacing: 0.5px;">
                {badge_text}
            </div>
            <div style="font-size: 0.85rem; font-weight: 600; color: {title_color}; margin: 4px 0 2px 0; line-height: 1.2;">
                {title}
            </div>
            <div style="font-size: 0.68rem; color: {desc_color}; line-height: 1.2;">
                {desc}
            </div>
        </div>
        """
        card_elements.append(card_html)

    full_grid = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; justify-content: flex-start;">
        {''.join(card_elements)}
    </div>
    """
    render_html_safe(full_grid)

    # Section 24 Architecture Overview Expander
    with st.expander("🏛️ **Bagan Blok Arsitektur Utuh (Section 24 Specification)**", expanded=False):
        block_elements = []
        for idx, (block_name, step_list, block_desc) in enumerate(ARCH_BLOCKS):
            is_active_block = (current_step in step_list)
            if is_active_block:
                block_bg = "background: linear-gradient(90deg, #1565c0, #00acc1); color: #ffffff; border: 2px solid #00e5ff; font-weight: bold; transform: scale(1.02);"
                status_badge = "<span style='background: #00e5ff; color: #000; font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; margin-left: 8px;'>AKTIF DI LANGKAH INI</span>"
            else:
                block_bg = "background: #252538; color: #dcdce6; border: 1px solid #3d3d56;"
                status_badge = ""

            block_html = f"""
            <div style="max-width: 480px; margin: 0 auto; text-align: center;">
                <div style="padding: 10px 16px; border-radius: 8px; margin: 3px auto; {block_bg} font-size: 0.95rem; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                    <b>{block_name}</b> {status_badge}
                    <div style="font-size: 0.72rem; opacity: 0.85; margin-top: 2px;">{block_desc}</div>
                </div>
            </div>
            """
            block_elements.append(block_html)

            # Add downward arrow if not the last block
            if idx < len(ARCH_BLOCKS) - 1:
                arrow_color = "#00e5ff" if is_active_block else "#666688"
                arrow_html = f"""
                <div style="text-align: center; color: {arrow_color}; font-size: 1.1rem; line-height: 1; margin: 1px 0;">
                    ↓
                </div>
                """
                block_elements.append(arrow_html)

        render_html_safe("".join(block_elements))
