# Transformer Layer Visualizer — MVP

Aplikasi web interaktif untuk media pembelajaran yang memvisualisasikan **forward pass satu Transformer Encoder layer (Post-LN)** secara transparan, langkah-demi-langkah, dan berbasis nilai numerik menggunakan **pure NumPy**, **Streamlit**, dan **Plotly**.

---

## 🎯 Tujuan

Fokus utama aplikasi ini bukan performa prediksi bahasa, melainkan **transparansi seluruh proses komputasi, operasi aljabar linier, dan perubahan bentuk (shape) matriks** sepanjang satu layer Transformer encoder.

Setiap operasi matriks ($XW_Q, QK^T, \text{softmax}, AV, ZW_O, \text{LayerNorm}, \text{FFN}$) disajikan secara eksplisit lengkap dengan:
1. Nilai numerik matriks terformat.
2. Dimensi tensor (shape tracker).
3. Penjelasan pedagogis: **WHAT**, **WHY**, **SHAPE**, dan peran **ALJABAR**.
4. Mode drill-down perhitungan sel individual / kombinasi linier.
5. Heatmap visual bobot atensi (*attention weights*) interaktif.

---

## 🏛️ Arsitektur Post-LN

Aplikasi mengimplementasikan standar Transformer Encoder layer dengan arsitektur **Post-LN**:

```text
Input Text ("Saya makan nasi goreng")
   ↓
Tokenization & Token IDs
   ↓
Embedding Lookup E[ids]  (N × d_model)
   ↓
+ Positional Encoding PE (N × d_model)
   ↓
Matriks Input X          (N × d_model)
   ↓
Multi-Head Self-Attention (Q, K, V Projections, Scaled Dot-Product, Softmax, Concat, WO)
   ↓
+ Residual 1 (X + MHA(X))
   ↓
LayerNorm 1 (Per-token over d_model)
   ↓
Feed-Forward Network (FFN: GELU(Y1 W1 + b1) W2 + b2)
   ↓
+ Residual 2 (Y1 + FFN(Y1))
   ↓
LayerNorm 2 (Akhir)
   ↓
Final Layer Output Y2    (N × d_model)
```

---

## ⚙️ Parameter Default

| Parameter | Nilai Default | Penjelasan |
| :--- | :--- | :--- |
| **Input Text** | `"Saya makan nasi goreng"` | Kalimat uji coba |
| **Token Count** | `4` | `["Saya", "makan", "nasi", "goreng"]` |
| **d_model** | `8` | Dimensi fitur representasi token |
| **num_heads** | `2` | Jumlah attention head |
| **d_head** | `4` | Dimensi proyeksi per-head ($d_{model} = num\_heads \times d_{head}$) |
| **d_ff** | `16` | Dimensi tersembunyi Feed-Forward Network |
| **epsilon** | `1e-5` | Stabilitas numerik LayerNorm |
| **Activation** | `GELU` | Non-linearitas FFN |
| **Normalization** | `LayerNorm` | Normalisasi statistik per-token |
| **Positional** | `Sinusoidal` | Encoding posisi harmonik Vaswani et al. |
| **Seed** | `42` | Menjamin reproduktibilitas bobot |

---

## 📁 Struktur Project

```text
Transformers Illustration/
│
├── app.py                      # Aplikasi utama Streamlit dengan navigator 12 langkah & tabs
│
├── transformer/                # Engine komputasi numerik (Pure NumPy)
│   ├── __init__.py
│   ├── tokenizer.py            # Tokenizer kata & pemetaan vocabulary
│   ├── embedding.py            # Lookup embedding deterministik
│   ├── positional_encoding.py  # Sinusoidal Positional Encoding
│   ├── attention.py            # Multi-Head Attention, Scaled Dot-Product, Concat, WO
│   ├── layer_norm.py           # Per-token Layer Normalization over d_model
│   ├── ffn.py                  # Position-wise FFN dengan aktivasi GELU
│   └── transformer_block.py    # Orkestrasi lengkap forward pass & perekaman trace
│
├── visualization/              # Komponen visual UI
│   ├── __init__.py
│   ├── matrix_view.py          # Render kartu matriks & operasi side-by-side
│   ├── attention_heatmap.py    # Heatmap interaktif Plotly dengan hover tooltips
│   ├── flow_diagram.py         # Diagram alur arsitektur dengan penanda langkah aktif
│   └── calculation_view.py     # Breakdown aljabar & perkalian sel individual
│
├── utils/                      # Fungsi utilitas & diagnostik
│   ├── __init__.py
│   ├── formatting.py           # Pemformat presisi angka & LaTeX
│   └── validation.py           # Pengecekan otomatis sifat matematika & dimensi
│
├── tests/
│   └── test_transformer.py     # Unit test matematis otomatis
│
├── requirements.txt            # Dependensi paket Python
└── README.md                   # Dokumentasi proyek
```

---

## 🚀 Instalasi & Menjalankan Aplikasi

### 1. Prasyarat
- Python 3.10 atau yang lebih baru (didukung hingga Python 3.13)
- pip atau uv

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser pada alamat default: `http://localhost:8501`.

### 4. Menjalankan Unit Test Otomatis
```bash
python -m unittest discover tests
```

---

## 🌟 Fitur Utama

1. **12-Step Pipeline Navigator**: Berpindah maju/mundur dengan tombol `⬅️ Sebelumnya` dan `Berikutnya ➡️` atau melompat langsung ke tahap mana pun.
2. **Side-by-Side Matrix Multiplication**: Memvisualisasikan $A \times B = C$ secara bersisian dengan label token dan dimensi yang jelas.
3. **Interactive Plotly Heatmap**: Peta panas atensi dengan tooltip dinamis (`Query`, `Key`, `Weight`) dan meteran validasi penjumlahan baris ($\sum = 1.000$).
4. **Cell-Level Math Explorer**:
   - Ekspansi dot product sel perkalian matriks ($Q[i,j] = \sum_k X[i,k] W_Q[k,j]$).
   - Penjabaran skor atensi per pasangan kata ($q_{Saya} \cdot k_{nasi} / \sqrt{d_{head}}$).
   - Penjabaran numerik bertahap LayerNorm ($\mu$, $\sigma^2$, $\hat{x}$, $\gamma$, $\beta$, $y$).
   - Penjabaran aproksimasi fungsi aktivasi GELU.
5. **Matrix Inspection Panel**: Katalog global untuk menginspeksi lebih dari 33 matriks individual yang diproduksi oleh pipeline.
6. **Shape Tracker**: Tabel pelacak bentuk dimensi tensor di sepanjang perjalanan forward pass.
7. **Validasi Numerik Real-time**: Verifikasi otomatis sifat invarian matematika ($d_{model} = num\_heads \times d_{head}$, row sums softmax, dan statistik LayerNorm).
