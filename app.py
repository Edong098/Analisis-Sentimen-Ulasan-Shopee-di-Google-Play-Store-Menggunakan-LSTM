# pyrefly: ignore [missing-import]
import streamlit as st
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import re
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

# pyrefly: ignore [missing-import]
from nltk.tokenize import word_tokenize
# pyrefly: ignore [missing-import]
from nltk.corpus import stopwords
# pyrefly: ignore [missing-import]
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# pyrefly: ignore [missing-import]
import nltk

for resource in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        else:
            nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download(resource, quiet=True)


st.set_page_config(
    page_title="Analisis Sentimen Shopee - LSTM",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim,
                 n_layers=2, dropout=0.3, fc_dropout=0.4):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,
            padding_idx=0
        )
        embedding_dim = embedding_matrix.shape[1]
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        embedded = self.embedding_dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]
        out = self.fc1(last_hidden)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# LOAD MODEL & DATA (cached)
base_dir = os.path.dirname(__file__)
output_dir = base_dir

@st.cache_resource
def load_model():
    """Load trained LSTM model."""
    matrix_path = os.path.join(output_dir, 'embedding_matrix.npy')
    embedding_matrix = np.load(matrix_path)
    model = SentimentLSTM(
        embedding_matrix=embedding_matrix,
        hidden_dim=128,
        output_dim=3,
        n_layers=2,
        dropout=0.3,
        fc_dropout=0.4
    )
    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, 'best_model_lexicon.pth'),
            map_location=torch.device('cpu')
        )
    )
    model.eval()
    return model, 100

@st.cache_resource
def load_tokenizer():
    """Load tokenizer data."""
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_eval_results():
    """Load evaluation results."""
    with open(os.path.join(output_dir, 'eval_results.json'), 'r') as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    """Load preprocessed dataset."""
    return pd.read_csv(os.path.join(base_dir, 'dataset_preprocessed.csv'))

@st.cache_resource
def load_stemmer():
    """Load Sastrawi stemmer only if needed."""
    try:
        with open(os.path.join(output_dir, 'tokenizer.pkl'), 'rb') as f:
            tok = pickle.load(f)
            if not tok.get('use_stemming', False):
                return None
    except Exception as e:
        pass
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_slang_dict():
    """Load slang words dictionary."""
    slang_path = os.path.join(base_dir, 'slangWords.json')
    if os.path.exists(slang_path):
        with open(slang_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# PREPROCESSING FUNCTIONS
def preprocess_text(text, tokenizer_data, stemmer, slang_dict=None):
    """
    Preprocess teks input untuk prediksi.
    Pipeline: clean -> lowercase -> slang fix -> tokenize -> stopword removal -> stemming
    """
    # 1. Cleaning
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Case folding
    text = text.lower()

    # 2b. Slang word normalization
    if slang_dict:
        words = text.split()
        fixed_words = [slang_dict.get(w, w) for w in words]
        text = ' '.join(fixed_words)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Stopword removal
    stop_words = {
        'di', 'ke', 'dari', 'pada', 'dalam', 'oleh', 'untuk',
        'dengan', 'antara', 'tentang', 'kepada', 'terhadap',
        'dan', 'atau', 'serta', 'yaitu', 'yakni', 'bahwa',
        'kita', 'mereka', 'ini', 'itu',
        'yang', 'lah', 'pun', 'per',
        'apa', 'siapa', 'mana', 'kapan',
        'nya', 'yg', 'dong', 'sih', 'loh', 'kok', 'tuh',
        'nih', 'nah', 'kan', 'deh', 'bgt',
        'gak', 'ga', 'gk', 'tp', 'krn', 'dgn', 'utk', 'jg', 'sm',
        'kalo', 'klo', 'klw', 'klu', 'gmn', 'emg', 'emang',
        'udh', 'sdh', 'blm', 'blum', 'dr', 'dri', 'pd', 'pda',
    }
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]

    # 5. Stemming (hanya jika model ditraining menggunakan stemming)
    use_stemming = tokenizer_data.get('use_stemming', False)
    if use_stemming and stemmer is not None:
        tokens = [stemmer.stem(w) for w in tokens]

    # 6. Text to sequence
    word_to_index = tokenizer_data['word_to_index']
    sequence = [word_to_index[w] for w in tokens if w in word_to_index]

    # 7. Padding
    max_len = tokenizer_data['max_length']
    if len(sequence) >= max_len:
        sequence = sequence[:max_len]
    else:
        sequence = [0] * (max_len - len(sequence)) + sequence

    return sequence, ' '.join(tokens)

def predict_sentiment(text, model, tokenizer_data, stemmer, max_length, slang_dict=None):
    """Predict sentiment of input text."""
    sequence, processed_text = preprocess_text(text, tokenizer_data, stemmer, slang_dict)
    input_tensor = torch.LongTensor([sequence])

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        predicted_class = np.argmax(probabilities)

    reverse_mapping = tokenizer_data['reverse_label_mapping']
    sentiment = reverse_mapping[predicted_class]

    return sentiment, probabilities, processed_text

# SIDEBAR NAVIGATION
st.sidebar.title("🛒 Shopee Sentiment")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["Dashboard", "Prediksi Sentimen", "Evaluasi Model", "Tentang Project"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** LSTM (Long Short Term Memory) & Word Embedding")
st.sidebar.markdown("**Framework:** PyTorch")
st.sidebar.markdown("**Dataset:** 30.000 ulasan Shopee")

# ============================================================
# HALAMAN 1: DASHBOARD
# ============================================================
if page == "Dashboard":
    st.title("📊 Analisis Sentimen Ulasan Aplikasi Shopee di Google Play Store")
    st.markdown("---")

    try:
        df = load_dataset()
        eval_results = load_eval_results()

        # Statistik Dataset
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ulasan", f"{len(df):,}")
        with col2:
            positif_count = len(df[df['sentiment'] == 'positif'])
            st.metric("Positif 😊", f"{positif_count:,}")
        with col3:
            netral_count = len(df[df['sentiment'] == 'netral'])
            st.metric("Netral 😐", f"{netral_count:,}")
        with col4:
            negatif_count = len(df[df['sentiment'] == 'negatif'])
            st.metric("Negatif 😠", f"{negatif_count:,}")

        st.markdown("---")

        # Grafik
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Sentimen")
            img_path = os.path.join(output_dir, 'distribusi_sentimen.png')
            if os.path.exists(img_path):
                st.image(img_path, width="stretch")

        with col2:
            st.subheader("Distribusi Rating")
            img_path = os.path.join(output_dir, 'distribusi_rating.png')
            if os.path.exists(img_path):
                st.image(img_path, width="stretch")

        st.markdown("---")

        # WordCloud
        st.subheader("WordCloud per Sentimen")
        img_path = os.path.join(output_dir, 'wordcloud_sentimen.png')
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")

        # Word Frequency
        st.subheader("Top Kata per Sentimen")
        img_path = os.path.join(output_dir, 'top_kata_per_sentimen.png')
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")

        # Word Count
        st.subheader("Distribusi Panjang Review")
        img_path = os.path.join(output_dir, 'word_count.png')
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Pastikan sudah menjalankan preprocessing.py, eda_feature_extraction.py, dan model_lstm.py terlebih dahulu.")

# ============================================================
# HALAMAN 2: PREDIKSI SENTIMEN
# ============================================================
elif page == "Prediksi Sentimen":
    st.title("Prediksi Sentimen Ulasan")
    st.markdown("Masukkan ulasan dalam Bahasa Indonesia dan model LSTM akan memprediksi sentimennya.")
    st.markdown("---")

    try:
        model, max_length = load_model()
        tokenizer_data = load_tokenizer()
        stemmer = load_stemmer()
        slang_dict = load_slang_dict()

        # Input text
        input_text = st.text_area(
            "Masukkan ulasan Shopee:",
            placeholder="Contoh: Aplikasi ini bagus banget, pengiriman cepat dan barang sesuai deskripsi!",
            height=120
        )

        # Contoh ulasan
        st.markdown("**Contoh ulasan untuk dicoba:**")
        examples = {
            "Positif 😊": "Aplikasi ini bagus banget, pengiriman cepat dan barang sesuai deskripsi! Suka belanja di shopee",
            "Netral 😐": "Aplikasi biasa saja, ada kelebihan dan kekurangannya. Lumayan lah untuk belanja online",
            "Negatif 😠": "Aplikasi jelek, sering error dan lambat. Barang tidak sesuai dan pengiriman lama sekali. Kecewa!"
        }

        cols = st.columns(3)
        for i, (label, text) in enumerate(examples.items()):
            with cols[i]:
                if st.button(label, key=f"example_{i}"):
                    input_text = text

        if st.button("🔍 Prediksi Sentimen", type="primary", width="stretch"):
            if input_text.strip():
                with st.spinner("Memproses..."):
                    sentiment, probs, processed = predict_sentiment(
                        input_text, model, tokenizer_data, stemmer, max_length, slang_dict
                    )

                st.markdown("---")

                # Hasil prediksi
                emoji_map = {'positif': '😊', 'negatif': '😠', 'netral': '😐'}
                color_map = {'positif': 'green', 'negatif': 'red', 'netral': 'orange'}

                st.markdown(f"### Hasil Prediksi: :{color_map[sentiment]}[**{sentiment.upper()}**] {emoji_map[sentiment]}")

                # Probabilitas
                st.markdown("#### Probabilitas per Kelas:")
                col1, col2, col3 = st.columns(3)

                labels = ['Negatif', 'Netral', 'Positif']
                colors = ['#e74c3c', '#f39c12', '#2ecc71']
                emojis = ['😠', '😐', '😊']

                for i, (col, label, prob, color, emoji) in enumerate(
                    zip([col1, col2, col3], labels, probs, colors, emojis)):
                    with col:
                        st.metric(f"{emoji} {label}", f"{prob*100:.1f}%")
                        st.progress(float(prob))

                # Detail preprocessing
                with st.expander("📝 Detail Preprocessing"):
                    st.markdown(f"**Teks asli:** {input_text}")
                    st.markdown(f"**Teks setelah preprocessing:** {processed}")

            else:
                st.warning("Silakan masukkan teks ulasan terlebih dahulu.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Pastikan model sudah di-training dengan menjalankan model_lstm.py terlebih dahulu.")

# ============================================================
# HALAMAN 3: EVALUASI MODEL
# ============================================================
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model LSTM")
    st.markdown("Hasil evaluasi performa model pada data testing.")
    st.markdown("---")

    try:
        eval_results = load_eval_results()

        # Metrik utama
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{eval_results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{eval_results['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{eval_results['recall']*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{eval_results['f1_score']*100:.2f}%")

        st.markdown("---")

        # Training History
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training History")
            img_path = os.path.join(output_dir, 'training_history.png')
            if os.path.exists(img_path):
                st.image(img_path, width="stretch")

        with col2:
            st.subheader("Confusion Matrix")
            img_path = os.path.join(output_dir, 'confusion_matrix.png')
            if os.path.exists(img_path):
                st.image(img_path, width="stretch")

        st.markdown("---")

        # Classification Report
        st.subheader("Classification Report")
        report = eval_results['classification_report']

        report_df = pd.DataFrame({
            'Kelas': ['Negatif', 'Netral', 'Positif'],
            'Precision': [
                report['Negatif']['precision'],
                report['Netral']['precision'],
                report['Positif']['precision']
            ],
            'Recall': [
                report['Negatif']['recall'],
                report['Netral']['recall'],
                report['Positif']['recall']
            ],
            'F1-Score': [
                report['Negatif']['f1-score'],
                report['Netral']['f1-score'],
                report['Positif']['f1-score']
            ],
            'Support': [
                int(report['Negatif']['support']),
                int(report['Netral']['support']),
                int(report['Positif']['support'])
            ]
        })

        st.dataframe(report_df, width="stretch", hide_index=True)

        st.markdown("---")

        # Hyperparameters
        st.subheader("Hyperparameters Model")
        hp = eval_results['hyperparameters']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            | Parameter | Nilai |
            |---|---|
            | Embedding Dim | {hp['embedding_dim']} |
            | Hidden Dim | {hp['hidden_dim']} |
            | LSTM Layers | {hp['n_layers']} |
            | Dropout | {hp['dropout']} |
            """)
        with col2:
            st.markdown(f"""
            | Parameter | Nilai |
            |---|---|
            | Learning Rate | {hp['learning_rate']} |
            | Batch Size | {hp['batch_size']} |
            | Epochs | {hp['num_epochs']} |
            | Max Length | {hp['max_length']} |
            """)

        # Analisis Performa
        st.markdown("---")
        st.subheader("Analisis Performa")

        final_train = eval_results['final_train_acc']
        final_val = eval_results['final_val_acc']
        gap = abs(final_train - final_val)

        if gap > 0.1:
            st.warning(f"""
            **Kemungkinan Overfitting Terdeteksi**
            - Training Accuracy: {final_train:.4f}
            - Validation Accuracy: {final_val:.4f}
            - Gap: {gap:.4f}

            Model mungkin terlalu 'menghafal' data training.
            """)
        elif final_train < 0.6:
            st.warning(f"""
            **Kemungkinan Underfitting Terdeteksi**
            - Training Accuracy: {final_train:.4f}
            - Validation Accuracy: {final_val:.4f}

            Model belum cukup belajar pola dari data.
            """)
        else:
            st.success(f"""
            **Model Memiliki Performa Baik!**
            - Training Accuracy: {final_train:.4f}
            - Validation Accuracy: {final_val:.4f}
            - Test Accuracy: {eval_results['accuracy']:.4f}
            - Gap: {gap:.4f}

            Model tidak menunjukkan tanda overfitting maupun underfitting.
            """)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Pastikan model sudah di-training dengan menjalankan model_lstm.py terlebih dahulu.")

# ============================================================
# HALAMAN 4: TENTANG PROJECT
# ============================================================
elif page == "Tentang Project":
    st.title("ℹ️ Tentang Project")
    st.markdown("---")

    st.markdown("""
    ## Analisis Sentimen Ulasan Pengguna Aplikasi Shopee di Google Play Store
    ### Menggunakan Model LSTM dan Word Embedding Berbasis Streamlit

    ---

    ### 📋 Deskripsi Project
    Project ini membangun sistem analisis sentimen berbasis web yang mampu
    mengklasifikasikan ulasan pengguna aplikasi Shopee di Google Play Store
    menjadi sentimen **Positif**, **Negatif**, dan **Netral** menggunakan
    metode Deep Learning **LSTM (Long Short-Term Memory)** dan **Word Embedding**.

    ---

    ### 🎯 Tujuan
    1. Menganalisis sentimen ulasan pengguna Shopee secara otomatis
    2. Membangun model deep learning LSTM untuk klasifikasi sentimen
    3. Membuat sistem prediksi sentimen berbasis web dengan Streamlit

    ---

    ### 📊 Dataset
    - **Sumber:** Google Play Store (Shopee Indonesia)
    - **Jumlah:** 30.000 ulasan
    - **Kolom:** Review (teks ulasan), Score (rating 1-5)
    - **Label:** Positif (rating 4-5), Netral (rating 3), Negatif (rating 1-2)

    ---

    ### 🤖 Model
    - **Arsitektur:** LSTM (Long Short-Term Memory)
    - **Framework:** PyTorch
    - **Word Embedding:** Trainable embedding layer
    - **Layers:**
        - Embedding Layer (vocab_size x 128)
        - LSTM Layer (2 layers, hidden_dim=128)
        - Dense Layer (64 neurons, ReLU)
        - Output Layer (3 classes, Softmax)

    ---

    ### 🔄 Pipeline NLP
    1. **Data Collection** - Scraping ulasan dari Google Play Store
    2. **Text Preprocessing** - Cleaning, tokenization, stopword removal, stemming
    3. **EDA** - Analisis distribusi sentimen, word frequency, wordcloud
    4. **Feature Extraction** - Tokenizer, text to sequence, padding
    5. **Model Training** - LSTM training dengan PyTorch
    6. **Evaluasi** - Accuracy, precision, recall, F1-score, confusion matrix
    7. **Deployment** - Streamlit web application

    ---

    ### 👥 Anggota Kelompok

    | No | Nama | Tugas |
    |---|---|---|
    | 1 | Ketua | Akuisisi Dataset, Integrasi, Dokumentasi | Streamlit web | 
    | 2 | Haura | Text Preprocessing |
    | 3 | Gulwani | EDA |
    | 4 | Jul | Feature Extraction |
    | 5 | Suta | Model LSTM |

    ---

    ### 🛠️ Teknologi
    - Python 3.14
    - PyTorch 2.12
    - Streamlit
    - NLTK & Sastrawi
    - scikit-learn
    - Matplotlib & Seaborn
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Mata Kuliah: Natural Language Processing*")
st.sidebar.markdown("*UAS - 2026*")
