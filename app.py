"""Transformer Layer Visualizer — MVP
Web application for transparent, step-by-step visual exploration of a 1-layer Post-LN Transformer Encoder.
"""

from typing import List, Tuple
import numpy as np
import streamlit as st

from transformer import (
    SimpleTokenizer,
    EmbeddingLayer,
    PositionalEncoding,
    MultiHeadAttention,
    LayerNorm,
    FeedForwardNetwork,
    TransformerBlock,
    PipelineTrace,
)
from visualization import (
    render_matrix_card,
    render_matrix_multiplication,
    render_attention_heatmap,
    render_matmul_element_detail,
    render_attention_score_detail,
    render_layernorm_token_detail,
    render_gelu_element_detail,
    render_pipeline_flow,
)
from utils import format_matrix_df, format_num, validate_pipeline, render_edu_card, render_html_safe

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transformer Layer Visualizer — MVP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern typography and clean presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e88e5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555577;
        margin-bottom: 1.5rem;
    }
    .edu-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1e88e5;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    @media (prefers-color-scheme: dark) {
        .edu-card {
            background-color: #1a1e29;
            border-left: 5px solid #00d2ff;
        }
        .sub-title {
            color: #aaaacc;
        }
    }
    .nav-box {
        background: #f0f4f8;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Parameters & Settings (Section 4 & 29)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Parameter")

input_text = st.sidebar.text_input(
    "Input Text",
    value="Saya makan nasi goreng",
    help="Teks kalimat input untuk diproses forward pass.",
)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    d_model = st.number_input("d_model", min_value=2, max_value=64, value=8, step=2)
    num_heads = st.number_input("num_heads", min_value=1, max_value=8, value=2, step=1)
with col_p2:
    d_head = st.number_input("d_head", min_value=1, max_value=32, value=4, step=1)
    d_ff = st.number_input("d_ff", min_value=4, max_value=128, value=16, step=4)

epsilon = st.sidebar.number_input("epsilon (LayerNorm)", value=1e-5, format="%.1e")
seed = st.sidebar.number_input("Random Seed (Reproducible)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Tampilan & Presisi")
precision = st.sidebar.slider("Desimal Presisi", min_value=1, max_value=6, value=3)

# Validate d_model == num_heads * d_head (Section 4)
if d_model != num_heads * d_head:
    st.error(
        f"❌ **Parameter Tidak Valid!**\n\n"
        f"`d_model` ({d_model}) harus sama dengan `num_heads` ({num_heads}) × `d_head` ({d_head}) = {num_heads * d_head}.\n"
        f"Forward pass dihentikan sampai kombinasi parameter diperbaiki."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Execute Transformer Numerical Forward Pass
# -----------------------------------------------------------------------------
try:
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        d_ff=d_ff,
        epsilon=epsilon,
        seed=seed,
    )
    trace = block.forward(input_text)
except Exception as e:
    st.error(f"Error saat mengeksekusi pipeline: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Session State for Step Navigation (Section 23)
# -----------------------------------------------------------------------------
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

TOTAL_STEPS = 12

STEP_NAMES = [
    "1. Input Text & Tokenization",
    "2. Embedding Matrix & Lookup",
    "3. Sinusoidal Positional Encoding",
    "4. Input Matrix X (Embedding + PE)",
    "5. Proyeksi Q, K, V (Head 1 & 2)",
    "6. Scaled Dot-Product Attention Scores",
    "7. Softmax & Attention Weights (Heatmap)",
    "8. Head Outputs (Z) & Concatenation",
    "9. Output Projection (WO) & MHA Output",
    "10. Residual 1 & LayerNorm 1",
    "11. Feed-Forward Network & GELU",
    "12. Residual 2 & LayerNorm Akhir (Output)",
]

# Title banner
st.markdown('<div class="main-title">🧠 Transformer Layer Visualizer — MVP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Transparansi perhitungan langkah-demi-langkah forward pass satu Transformer Encoder Layer (Post-LN) menggunakan pure NumPy.</div>',
    unsafe_allow_html=True,
)

# Render Global Flow Diagram (Section 24)
render_pipeline_flow(st.session_state.current_step)

# -----------------------------------------------------------------------------
# Top Navigation Bar (Section 23)
# -----------------------------------------------------------------------------
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 4, 2, 4])

with nav_col1:
    if st.button("⬅️ Sebelumnya", disabled=(st.session_state.current_step <= 1), use_container_width=True):
        st.session_state.current_step -= 1
        st.rerun()

with nav_col2:
    step_selected = st.selectbox(
        "Pilih Langkah",
        options=list(range(1, TOTAL_STEPS + 1)),
        format_func=lambda x: STEP_NAMES[x - 1],
        index=st.session_state.current_step - 1,
        label_visibility="collapsed",
    )
    if step_selected != st.session_state.current_step:
        st.session_state.current_step = step_selected
        st.rerun()

with nav_col3:
    if st.button("Berikutnya ➡️", disabled=(st.session_state.current_step >= TOTAL_STEPS), use_container_width=True):
        st.session_state.current_step += 1
        st.rerun()

with nav_col4:
    st.progress(st.session_state.current_step / TOTAL_STEPS, text=f"Langkah {st.session_state.current_step} dari {TOTAL_STEPS}")

# -----------------------------------------------------------------------------
# Main Tabs: Step View, Matrix Inspector, Shape Tracker, Numerical Validation
# -----------------------------------------------------------------------------
tab_walkthrough, tab_matrix_inspector, tab_shapes, tab_validation = st.tabs(
    ["📖 Penelusuran Langkah", "🔍 Matrix Inspection Panel", "📐 Shape Tracker", "✅ Validasi Numerik"]
)

# =============================================================================
# TAB 1: STEP WALKTHROUGH
# =============================================================================
with tab_walkthrough:
    cur = st.session_state.current_step
    tokens = trace.tokens
    token_ids = trace.token_ids
    n_tokens = len(tokens)
    token_labels = [f"pos{i}:{tok}" for i, tok in enumerate(tokens)]
    dim_labels = [f"d{j}" for j in range(d_model)]
    head_dim_labels = [f"h{j}" for j in range(d_head)]
    ffn_dim_labels = [f"ff{j}" for j in range(d_ff)]

    # -------------------------------------------------------------------------
    # STEP 1: Input & Tokens (Section 5)
    # -------------------------------------------------------------------------
    if cur == 1:
        st.subheader("Langkah 1: Input Text & Tokenization")
        render_edu_card(
            what="Memecah kalimat input teks menjadi token kata diskrit, lalu memetakannya ke Token ID integer melalui vocabulary.",
            why="Komputer dan jaringan neural memerlukan input berupa representasi angka integer/vektor, bukan karakter teks mentah.",
            shape="String -> List[str] (panjang N) -> List[int] (panjang N)",
            aljabar="Pemetaan diskrit / Integer Lookup.",
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### 📝 Token yang Dihasilkan")
            token_df = {
                "Posisi": list(range(n_tokens)),
                "Token": tokens,
                "Token ID": token_ids,
            }
            st.table(token_df)

        with col_t2:
            st.markdown("#### 📚 Vocabulary Lookup Table")
            vocab_items = list(trace.vocab.items())
            st.dataframe(
                {"Token": [k for k, _ in vocab_items], "ID": [v for _, v in vocab_items]},
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # STEP 2: Embedding (Section 6)
    # -------------------------------------------------------------------------
    elif cur == 2:
        st.subheader("Langkah 2: Embedding Matrix & Token Lookup")
        render_edu_card(
            what="Mengambil vektor berdimensi <code>d_model</code> dari matriks tabel embedding <code>E</code> untuk setiap token ID.",
            why="Memberikan representasi awal fitur semantik pada ruang berdimensi kontinu untuk setiap kata.",
            shape="E ∈ R^(vocab_size × d_model), X_embed = E[token_ids] ∈ R^(N × d_model)",
            aljabar="Table Indexing / One-hot Matrix Multiplication (<code>X_embed = OneHot @ E</code>).",
        )

        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            render_matrix_card(
                trace.E,
                title="Embedding Table E",
                algebraic_label="Learnable Embedding Dictionary",
                row_labels=[f"ID {v} ({k})" for k, v in trace.vocab.items() if v < trace.E.shape[0]],
                col_labels=dim_labels,
                precision=precision,
                note=f"Tabel embedding berisi vektor untuk seluruh vocabulary ({trace.E.shape[0]} kata).",
            )
        with col_e2:
            render_matrix_card(
                trace.X_embed,
                title="X_embed (Hasil Lookup)",
                algebraic_label="Selected Token Embeddings E[token_ids]",
                row_labels=token_labels,
                col_labels=dim_labels,
                precision=precision,
                note=f"Setiap baris adalah vektor d_model={d_model} untuk token yang ada di kalimat.",
            )

    # -------------------------------------------------------------------------
    # STEP 3: Positional Encoding (Section 7)
    # -------------------------------------------------------------------------
    elif cur == 3:
        st.subheader("Langkah 3: Sinusoidal Positional Encoding")
        render_edu_card(
            what="Menghitung matriks encoding posisi menggunakan kombinasi gelombang sinus dan kosinus dengan frekuensi berbeda.",
            why="Arsitektur Transformer memproses semua token secara paralel (tanpa rekurensi RNN). Positional encoding memberikan informasi urutan/posisi tanpa mengubah dimensi fitur.",
            shape="PE ∈ R^(N × d_model)",
            aljabar="Evaluasi fungsi harmonik sinusoidal terhadap posisi & indeks dimensi.",
        )

        st.latex(
            r"PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)"
        )

        render_matrix_card(
            trace.PE,
            title="Positional Encoding Matrix PE",
            algebraic_label="Deterministic Harmonic Vectors",
            row_labels=[f"pos {i} ({tok})" for i, tok in enumerate(tokens)],
            col_labels=dim_labels,
            precision=precision,
            note="Perhatikan bahwa dimensi genap menggunakan sin(pos/...) dan ganjil menggunakan cos(pos/...).",
        )

    # -------------------------------------------------------------------------
    # STEP 4: Input Matrix X (Section 7)
    # -------------------------------------------------------------------------
    elif cur == 4:
        st.subheader("Langkah 4: Matriks Input X = Embedding + Positional Encoding")
        render_edu_card(
            what="Menjumlahkan secara element-wise matriks embedding token dengan matriks positional encoding.",
            why="Menggabungkan identitas semantik kata dengan letak posisinya dalam satu representasi terpadu.",
            shape="(N × d_model) + (N × d_model) = (N × d_model)",
            aljabar="Element-wise Matrix Addition.",
        )

        render_matrix_multiplication(
            trace.X_embed,
            "X_embed",
            trace.PE,
            "PE",
            trace.X,
            "Input Matrix X",
            op_symbol="+",
            algebraic_label="Element-wise Addition (X = X_embed + PE)",
            rows_a=token_labels,
            cols_a=dim_labels,
            rows_b=token_labels,
            cols_b=dim_labels,
            rows_res=token_labels,
            cols_res=dim_labels,
            precision=precision,
        )

    # -------------------------------------------------------------------------
    # STEP 5: Q, K, V Projections (Sections 8, 9, 10, 38)
    # -------------------------------------------------------------------------
    elif cur == 5:
        st.subheader("Langkah 5: Linear Projections Q, K, dan V")
        render_edu_card(
            what="Mengalikan matriks input <code>X</code> dengan tiga set bobot terpisah (<code>WQ</code>, <code>WK</code>, <code>WV</code>) untuk setiap head.",
            why="Memetakan <code>X</code> ke ruang representasi khusus: Query (apa yang dicari), Key (apa yang ditawarkan), dan Value (informasi konten). <b>Prinsip:</b> <code>X</code> tidak dibagi!",
            shape="(N × d_model) @ (d_model × d_head) = (N × d_head)",
            aljabar="Learned Linear Transformation / Matrix Multiplication.",
        )

        selected_head_idx = st.radio("Pilih Attention Head:", options=[1, 2], horizontal=True, format_func=lambda h: f"Head {h}")
        head_data = trace.attention_trace.heads[selected_head_idx - 1]

        proj_tab1, proj_tab2, proj_tab3 = st.tabs(["Query (Q)", "Key (K)", "Value (V)"])

        with proj_tab1:
            render_matrix_multiplication(
                trace.X,
                "X",
                head_data.W_Q,
                f"WQ{selected_head_idx}",
                head_data.Q,
                f"Q{selected_head_idx}",
                algebraic_label=f"Query Projection Head {selected_head_idx} (X @ WQ)",
                rows_a=token_labels,
                cols_a=dim_labels,
                rows_b=dim_labels,
                cols_b=head_dim_labels,
                rows_res=token_labels,
                cols_res=head_dim_labels,
                precision=precision,
            )

        with proj_tab2:
            render_matrix_multiplication(
                trace.X,
                "X",
                head_data.W_K,
                f"WK{selected_head_idx}",
                head_data.K,
                f"K{selected_head_idx}",
                algebraic_label=f"Key Projection Head {selected_head_idx} (X @ WK)",
                rows_a=token_labels,
                cols_a=dim_labels,
                rows_b=dim_labels,
                cols_b=head_dim_labels,
                rows_res=token_labels,
                cols_res=head_dim_labels,
                precision=precision,
            )

        with proj_tab3:
            render_matrix_multiplication(
                trace.X,
                "X",
                head_data.W_V,
                f"WV{selected_head_idx}",
                head_data.V,
                f"V{selected_head_idx}",
                algebraic_label=f"Value Projection Head {selected_head_idx} (X @ WV)",
                rows_a=token_labels,
                cols_a=dim_labels,
                rows_b=dim_labels,
                cols_b=head_dim_labels,
                rows_res=token_labels,
                cols_res=head_dim_labels,
                precision=precision,
            )

        # Detailed Element Inspection (Section 10)
        st.markdown("---")
        with st.expander("🔍 **Mode Penjelasan Perhitungan Satu Elemen (Section 10)**", expanded=False):
            st.write("Pilih sel elemen pada matriks Query (Q) untuk melihat perkalian baris X dan kolom WQ:")
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                sel_row = st.selectbox("Baris (Token)", options=list(range(n_tokens)), format_func=lambda i: f"Baris {i} ('{tokens[i]}')")
            with col_sel2:
                sel_col = st.selectbox("Kolom (Fitur Head)", options=list(range(d_head)), format_func=lambda j: f"Kolom {j}")

            render_matmul_element_detail(
                trace.X,
                "X",
                head_data.W_Q,
                f"WQ{selected_head_idx}",
                head_data.Q,
                f"Q{selected_head_idx}",
                row_idx=sel_row,
                col_idx=sel_col,
                row_label=tokens[sel_row],
                col_label=f"h{sel_col}",
                precision=precision,
            )

    # -------------------------------------------------------------------------
    # STEP 6: Scaled Dot-Product Scores (Sections 11 & 13)
    # -------------------------------------------------------------------------
    elif cur == 6:
        st.subheader("Langkah 6: Scaled Dot-Product Attention Scores")
        render_edu_card(
            what="Menghitung dot product antara setiap vektor Query dengan seluruh vektor Key yang ditransposisikan (<code>Q @ K.T</code>), lalu membaginya dengan <code>sqrt(d_head)</code>.",
            why="Mengukur kecocokan/afinitas keterkaitan antara setiap pasang token. Penskalaan dengan <code>sqrt(d_head)</code> mencegah magnitude skor menjadi terlalu besar yang dapat menyebabkan gradien softmax vanishing.",
            shape="(N × d_head) @ (d_head × N) = (N × N)",
            aljabar="Dot Product Affinity Scoring / Bilinear Similarity.",
        )

        selected_head_idx = st.radio("Pilih Attention Head:", options=[1, 2], horizontal=True, format_func=lambda h: f"Head {h}")
        head_data = trace.attention_trace.heads[selected_head_idx - 1]

        render_matrix_multiplication(
            head_data.Q,
            f"Q{selected_head_idx}",
            head_data.K.T,
            f"K{selected_head_idx}^T",
            head_data.scores,
            f"Scores{selected_head_idx}",
            algebraic_label=f"Scores = (Q @ K^T) / sqrt({d_head})",
            rows_a=token_labels,
            cols_a=head_dim_labels,
            rows_b=head_dim_labels,
            cols_b=token_labels,
            rows_res=token_labels,
            cols_res=token_labels,
            precision=precision,
        )

        # Concept explanation on attention score (Section 13)
        st.markdown("---")
        with st.expander("🔍 **Detail Perhitungan Alignment Score Antar Token (Section 13)**", expanded=True):
            st.write("Pilih pasangan Query token dan Key token untuk melihat dot product dan penskalaannya:")
            c_q, c_k = st.columns(2)
            with c_q:
                q_idx = st.selectbox("Query Token (Q)", options=list(range(n_tokens)), format_func=lambda i: tokens[i], index=0)
            with c_k:
                k_idx = st.selectbox("Key Token (K)", options=list(range(n_tokens)), format_func=lambda i: tokens[i], index=min(2, n_tokens - 1))

            render_attention_score_detail(
                head_data.Q,
                head_data.K,
                tokens=tokens,
                q_idx=q_idx,
                k_idx=k_idx,
                d_head=d_head,
                precision=precision,
            )

    # -------------------------------------------------------------------------
    # STEP 7: Softmax & Attention Weights (Section 12)
    # -------------------------------------------------------------------------
    elif cur == 7:
        st.subheader("Langkah 7: Softmax & Attention Weights Heatmap")
        render_edu_card(
            what="Menerapkan fungsi Softmax secara horizontal (per baris) pada matriks skor perhatian.",
            why="Mengubah skor kecocokan menjadi distribusi probabilitas berbobot antara 0 dan 1, di mana setiap baris berjumlah 1.0. Bobot ini menentukan seberapa besar perhatian query kepada masing-masing key.",
            shape="A = softmax(scores, axis=-1) ∈ R^(N × N)",
            aljabar="Probability Normalization.",
        )

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            head_1 = trace.attention_trace.heads[0]
            render_attention_heatmap(head_1.attention_weights, tokens=tokens, head_title="Head 1", precision=precision)

        with col_h2:
            if len(trace.attention_trace.heads) > 1:
                head_2 = trace.attention_trace.heads[1]
                render_attention_heatmap(head_2.attention_weights, tokens=tokens, head_title="Head 2", precision=precision)

    # -------------------------------------------------------------------------
    # STEP 8: Head Outputs (Z) & Concatenation (Sections 11 & 14)
    # -------------------------------------------------------------------------
    elif cur == 8:
        st.subheader("Langkah 8: Weighted Aggregation (Z = A @ V) & Head Concatenation")
        render_edu_card(
            what="Untuk setiap head, kalikan attention weights <code>A</code> dengan matriks <code>V</code> untuk menghasilkan konteks <code>Z_i</code>, lalu gabungkan secara horizontal (concatenation).",
            why="Menghimpun informasi representasi dari seluruh token sesuai bobot relevansinya, dan menggabungkan berbagai perspektif pemahaman dari multi-head.",
            shape="Z_i = (N × N) @ (N × d_head) = (N × d_head), lalu Concat(Z1, Z2) = (N × d_model)",
            aljabar="Convex Combination / Weighted Averaging lalu Horizontal Tensor Concatenation.",
        )

        st.markdown("#### 1. Perhitungan Output Tiap Head: $Z_i = A_i \\times V_i$")
        z_tabs = st.tabs([f"Head {h.head_idx}" for h in trace.attention_trace.heads])
        for idx, h in enumerate(trace.attention_trace.heads):
            with z_tabs[idx]:
                render_matrix_multiplication(
                    h.attention_weights,
                    f"A{h.head_idx}",
                    h.V,
                    f"V{h.head_idx}",
                    h.Z,
                    f"Z{h.head_idx}",
                    algebraic_label=f"Context Vector Head {h.head_idx} (Weighted Average)",
                    rows_a=token_labels,
                    cols_a=token_labels,
                    rows_b=token_labels,
                    cols_b=head_dim_labels,
                    rows_res=token_labels,
                    cols_res=head_dim_labels,
                    precision=precision,
                )

        st.markdown("#### 2. Concatenation Antar-Head: $Z = [Z_1, Z_2, \\dots]$")
        render_matrix_card(
            trace.attention_trace.Z_concat,
            title="Concatenated Heads Z",
            algebraic_label="Horizontal Concatenation Concat(Z1, Z2)",
            row_labels=token_labels,
            col_labels=dim_labels,
            precision=precision,
            note=f"Perhatikan kolom [0..{d_head-1}] berasal dari Head 1 dan kolom [{d_head}..{d_model-1}] berasal dari Head 2.",
        )

    # -------------------------------------------------------------------------
    # STEP 9: Output Projection WO & MHA Output (Section 15)
    # -------------------------------------------------------------------------
    elif cur == 9:
        st.subheader("Langkah 9: Output Projection (WO) & MHA Output")
        render_edu_card(
            what="Mengalikan matriks gabungan <code>Z</code> dengan matriks bobot proyeksi <code>W_O</code>.",
            why="Mengintegrasikan fitur-fitur dari berbagai head kembali ke dalam representasi terpadu berdimensi <code>d_model</code>.",
            shape="(N × d_model) @ (d_model × d_model) = (N × d_model)",
            aljabar="Linear Synthesis Transformation.",
        )

        render_matrix_multiplication(
            trace.attention_trace.Z_concat,
            "Z (Concat)",
            trace.attention_trace.W_O,
            "WO",
            trace.attention_trace.MHA_output,
            "MHA(X)",
            algebraic_label="Final Multi-Head Attention Projection (Z @ WO)",
            rows_a=token_labels,
            cols_a=dim_labels,
            rows_b=dim_labels,
            cols_b=dim_labels,
            rows_res=token_labels,
            cols_res=dim_labels,
            precision=precision,
        )

        with st.expander("🔍 **Detail Perhitungan Satu Sel MHA[i, j]**", expanded=False):
            c_r, c_c = st.columns(2)
            with c_r:
                r_mha = st.selectbox("Baris (Token)", options=list(range(n_tokens)), format_func=lambda i: tokens[i], key="mha_r")
            with c_c:
                c_mha = st.selectbox("Kolom Fitur", options=list(range(d_model)), format_func=lambda j: f"d{j}", key="mha_c")

            render_matmul_element_detail(
                trace.attention_trace.Z_concat,
                "Z",
                trace.attention_trace.W_O,
                "WO",
                trace.attention_trace.MHA_output,
                "MHA",
                row_idx=r_mha,
                col_idx=c_mha,
                row_label=tokens[r_mha],
                col_label=f"d{c_mha}",
                precision=precision,
            )

    # -------------------------------------------------------------------------
    # STEP 10: Residual 1 & LayerNorm 1 (Sections 16, 17, 18)
    # -------------------------------------------------------------------------
    elif cur == 10:
        st.subheader("Langkah 10: Residual Connection 1 & Layer Normalization 1")
        render_edu_card(
            what="Menjumlahkan input <code>X</code> dengan output <code>MHA(X)</code> (Residual), lalu menormalisasi vektor setiap token secara independen (LayerNorm).",
            why="Residual connection mencegah degradasi informasi dan membantu kestabilan aliran gradien. LayerNorm menjaga rata-rata mendekati 0 dan variansi 1.",
            shape="R1 = X + MHA(X) ∈ R^(N × d_model), Y1 = LayerNorm(R1) ∈ R^(N × d_model)",
            aljabar="Skip-connection Addition lalu Per-token Statistical Normalization.",
        )

        st.markdown("#### 1. Residual Connection Pertama: $R_1 = X + \\text{MHA}(X)$")
        render_matrix_multiplication(
            trace.X,
            "Input X",
            trace.attention_trace.MHA_output,
            "MHA(X)",
            trace.R1,
            "Residual R1",
            op_symbol="+",
            algebraic_label="Element-wise Residual Addition",
            rows_a=token_labels,
            cols_a=dim_labels,
            rows_b=token_labels,
            cols_b=dim_labels,
            rows_res=token_labels,
            cols_res=dim_labels,
            precision=precision,
        )

        st.markdown("#### 2. Layer Normalization 1: $Y_1 = \\text{LayerNorm}(R_1)$")
        render_matrix_card(
            trace.Y1,
            title="Sublayer 1 Output Y1",
            algebraic_label="Normalized Output Y1",
            row_labels=token_labels,
            col_labels=dim_labels,
            precision=precision,
            note="Nilai output setelah dinormalisasi dengan gamma=1 dan beta=0.",
        )

        with st.expander("🔍 **Detail Perhitungan LayerNorm per Token (Section 18)**", expanded=True):
            tok_ln1_idx = st.selectbox(
                "Pilih Token untuk Ditelusuri Detail LayerNorm 1:",
                options=list(range(n_tokens)),
                format_func=lambda i: f"Token '{tokens[i]}' (Posisi {i})",
                key="ln1_tok",
            )
            render_layernorm_token_detail(
                trace.R1,
                token_label=tokens[tok_ln1_idx],
                token_idx=tok_ln1_idx,
                epsilon=epsilon,
                gamma=trace.ln1_trace.gamma,
                beta=trace.ln1_trace.beta,
                precision=precision,
            )

    # -------------------------------------------------------------------------
    # STEP 11: Feed-Forward Network & GELU (Sections 19 & 20)
    # -------------------------------------------------------------------------
    elif cur == 11:
        st.subheader("Langkah 11: Position-wise Feed-Forward Network & GELU Activation")
        render_edu_card(
            what="Memproses setiap token secara independen melalui dua lapisan linier dengan aktivasi non-linier GELU: <code>H_pre = Y1 W1 + b1</code>, <code>H = GELU(H_pre)</code>, <code>FFN = H W2 + b2</code>.",
            why="Mengembangkan kapasitas representasi dan mengekspresikan interaksi fitur non-linier. FFN diproses per token secara terpisah, tidak ada pertukaran informasi antar-token!",
            shape="Y1 (N × d_model) -> H (N × d_ff) -> FFN (N × d_model)",
            aljabar="Linear Expansion -> Non-linear Activation -> Linear Contraction.",
        )

        ffn_tab1, ffn_tab2, ffn_tab3 = st.tabs(["1. Lapisan Pertama (W1 & H_pre)", "2. Aktivasi GELU (H)", "3. Lapisan Kedua (W2 & Output FFN)"])

        with ffn_tab1:
            render_matrix_multiplication(
                trace.Y1,
                "Y1",
                trace.ffn_trace.W_1,
                "W1",
                trace.ffn_trace.H_pre,
                "H_pre",
                algebraic_label="FFN Expansion (Y1 @ W1)",
                rows_a=token_labels,
                cols_a=dim_labels,
                rows_b=dim_labels,
                cols_b=ffn_dim_labels,
                rows_res=token_labels,
                cols_res=ffn_dim_labels,
                precision=precision,
            )

        with ffn_tab2:
            render_matrix_card(
                trace.ffn_trace.H,
                title="Aktivasi H = GELU(H_pre)",
                algebraic_label="Smooth Non-linear Activation",
                row_labels=token_labels,
                col_labels=ffn_dim_labels,
                precision=precision,
                note="Nilai H_pre dipetakan melalui fungsi aktivasi GELU.",
            )
            with st.expander("🔍 **Detail Perhitungan Aktivasi GELU (Section 20)**", expanded=False):
                sample_val = float(trace.ffn_trace.H_pre[0, 0])
                input_x = st.number_input("Masukkan nilai x untuk dievaluasi GELU:", value=sample_val, format="%.4f")
                render_gelu_element_detail(input_x, precision=precision)

        with ffn_tab3:
            render_matrix_multiplication(
                trace.ffn_trace.H,
                "H (GELU)",
                trace.ffn_trace.W_2,
                "W2",
                trace.ffn_trace.FFN_output,
                "FFN Output",
                algebraic_label="FFN Projection Back to d_model (H @ W2)",
                rows_a=token_labels,
                cols_a=ffn_dim_labels,
                rows_b=ffn_dim_labels,
                cols_b=dim_labels,
                rows_res=token_labels,
                cols_res=dim_labels,
                precision=precision,
            )

    # -------------------------------------------------------------------------
    # STEP 12: Residual 2 & Final LayerNorm (Sections 21 & 22)
    # -------------------------------------------------------------------------
    elif cur == 12:
        st.subheader("Langkah 12: Residual Connection 2 & Final LayerNorm Output")
        render_edu_card(
            what="Menjumlahkan output sublayer pertama <code>Y1</code> dengan output sublayer FFN (Residual 2), lalu menerapkan LayerNorm akhir.",
            why="Menghasilkan representasi kontekstual akhir <code>Y2</code> dari satu lapisan Transformer Encoder yang siap diteruskan ke lapisan berikutnya.",
            shape="R2 = Y1 + FFN ∈ R^(N × d_model), Y2 = LayerNorm(R2) ∈ R^(N × d_model)",
            aljabar="Residual Addition lalu Per-token Normalization.",
        )

        st.markdown("#### 1. Residual Connection Kedua: $R_2 = Y_1 + \\text{FFN}(Y_1)$")
        render_matrix_multiplication(
            trace.Y1,
            "Y1",
            trace.ffn_trace.FFN_output,
            "FFN",
            trace.R2,
            "Residual R2",
            op_symbol="+",
            algebraic_label="Second Residual Addition (Y1 + FFN)",
            rows_a=token_labels,
            cols_a=dim_labels,
            rows_b=token_labels,
            cols_b=dim_labels,
            rows_res=token_labels,
            cols_res=dim_labels,
            precision=precision,
        )

        st.markdown("#### 2. LayerNorm Akhir: $Y_2 = \\text{LayerNorm}(R_2)$")
        render_matrix_card(
            trace.Y2,
            title="Output of Transformer Layer (Y2)",
            algebraic_label="Final Layer-Normalized Representation",
            row_labels=token_labels,
            col_labels=dim_labels,
            precision=precision,
            note="Representasi vektor akhir untuk seluruh token pada layer ini.",
        )

        with st.expander("🔍 **Detail Perhitungan LayerNorm 2 per Token**", expanded=False):
            tok_ln2_idx = st.selectbox(
                "Pilih Token untuk Ditelusuri Detail LayerNorm 2:",
                options=list(range(n_tokens)),
                format_func=lambda i: f"Token '{tokens[i]}' (Posisi {i})",
                key="ln2_tok",
            )
            render_layernorm_token_detail(
                trace.R2,
                token_label=tokens[tok_ln2_idx],
                token_idx=tok_ln2_idx,
                epsilon=epsilon,
                gamma=trace.ln2_trace.gamma,
                beta=trace.ln2_trace.beta,
                precision=precision,
            )

        st.success("🎉 **Forward pass satu Transformer Encoder layer selesai secara lengkap dan transparan!**")

# =============================================================================
# TAB 2: MATRIX INSPECTION PANEL (Section 25)
# =============================================================================
with tab_matrix_inspector:
    st.subheader("🔍 Matrix Inspection Panel")
    st.markdown(
        "Panel khusus untuk memeriksa nilai numerik, shape, dan deskripsi dari seluruh matriks yang diproduksi dalam pipeline."
    )

    mat_keys = list(trace.matrices_dict.keys())
    selected_mat_name = st.selectbox(
        "Pilih Matriks untuk Diinspeksi:",
        options=mat_keys,
        format_func=lambda k: f"{k} — {trace.matrices_dict[k][2]}",
        index=mat_keys.index("Y2") if "Y2" in mat_keys else 0,
    )

    mat_val, mat_shape, mat_desc = trace.matrices_dict[selected_mat_name]

    c_info1, c_info2 = st.columns([1, 2])
    with c_info1:
        st.metric("Nama Matriks", selected_mat_name)
        st.metric("Shape", f"{mat_shape}")
    with c_info2:
        st.info(f"📋 **Deskripsi:** {mat_desc}")

    # Display full values
    st.markdown("#### Nilai Matriks")
    df_inspect = format_matrix_df(mat_val, precision=precision)
    st.dataframe(df_inspect, use_container_width=True)

# =============================================================================
# TAB 3: SHAPE TRACKER (Section 26)
# =============================================================================
with tab_shapes:
    st.subheader("📐 Shape Tracker")
    st.markdown("Pelacakan dimensi tensor secara otomatis di setiap tahap eksekusi.")

    st.table(trace.shapes_summary)

# =============================================================================
# TAB 4: NUMERICAL VALIDATION (Section 32)
# =============================================================================
with tab_validation:
    st.subheader("✅ Validasi Numerik & Sanity Checks")
    st.markdown("Pengecekan otomatis sifat matematis dan konsistensi dimensi:")

    validation_items = validate_pipeline(trace, d_model=d_model, num_heads=num_heads, d_head=d_head, d_ff=d_ff)

    for item in validation_items:
        icon = "✅" if item.passed else "❌"
        status_text = "LULUS" if item.passed else "GAGAL"
        with st.expander(f"{icon} **[{item.category}]** {item.check_name} — *{status_text}*", expanded=True):
            st.write(f"**Detail:** {item.details}")
