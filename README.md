# 🛒 Analisis Sentimen Ulasan Shopee - LSTM & Word2Vec

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/DL-PyTorch-orange)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%26%20Sastrawi-green)](https://github.com/sastrawi/sastrawi)

Sistem Analisis Sentimen Ulasan Aplikasi Shopee di Google Play Store berbasis web interaktif. Aplikasi ini mengklasifikasikan ulasan pengguna menjadi tiga kategori sentimen (**Positif**, **Netral**, dan **Negatif**) menggunakan arsitektur Deep Learning **LSTM (Long Short-Term Memory)** dan pemodelan representasi kata **Word2Vec (Skip-gram)**.

---

## 🚀 Fitur Utama

1. **📊 Dashboard Interaktif**:
   * Statistik ringkasan ulasan (Total Ulasan, Positif, Netral, Negatif).
   * Grafik visualisasi data eksplorasi (EDA): Distribusi Sentimen, Distribusi Rating, WordCloud per Sentimen, Top Kata per Sentimen, dan Distribusi Panjang Review.
2. **🔍 Prediksi Sentimen**:
   * Input ulasan baru secara dinamis untuk diklasifikasikan oleh model LSTM.
   * Contoh ulasan sekali klik (Positif, Netral, Negatif) untuk pengujian cepat.
   * Visualisasi persentase probabilitas hasil prediksi untuk setiap kelas menggunakan progress bar.
   * Detail hasil *preprocessing* teks (teks bersih sebelum diinput ke model).
3. **📈 Evaluasi Performa Model**:
   * Metrik performa utama: *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.
   * Grafik *Training History* (Accuracy/Loss) dan *Confusion Matrix*.
   * Tabel detail *Classification Report* per kelas.
   * Parameter arsitektur model dan analisis performa (Overfitting/Underfitting).
4. **ℹ️ Tentang Project**:
   * Informasi lengkap mengenai dataset, arsitektur model LSTM, pipeline NLP, dan anggota kelompok.

---

## 🛠️ Arsitektur & Pipeline NLP

Sistem ini dibangun dengan pipeline pemrosesan bahasa alami (NLP) yang sistematis:
1. **Pembersihan Data (Text Cleaning)**: Menghapus emoji, username, hashtag, tautan url, angka, tanda baca, dan spasi berlebih.
2. **Case Folding**: Mengubah seluruh teks menjadi huruf kecil (lowercase).
3. **Normalisasi Slang**: Memperbaiki kata singkatan/slang menggunakan kamus `slangWords.json` (contoh: *bgt* $\rightarrow$ *banget*, *gpp* $\rightarrow$ *tidak apa-apa*).
4. **Tokenisasi**: Memotong teks ulasan menjadi daftar token kata.
5. **Stopword Removal**: Menghapus kata-kata tidak bermakna menggunakan stopwords kustom Bahasa Indonesia tanpa melakukan *stemming* guna mempertahankan makna kontekstual negasi.
6. **Ekstraksi Fitur**: Menggunakan representasi vektor kata **Word2Vec** berdimensi 128 (dilatih secara mandiri menggunakan Gensim Skip-gram) yang diintegrasikan sebagai *pretrained weight* pada PyTorch Embedding Layer.
7. **Klasifikasi LSTM**: Arsitektur jaringan saraf berulang (RNN) dengan:
   * **Embedding Layer**: $11910 \times 128$ (dengan parameter trainable disetel ke `True`).
   * **LSTM Layer**: 2 layer unidirectional (hidden dimension = 128, dropout = 0.3).
   * **Fully Connected Layer**: Dense layer (64 neuron dengan aktivasi ReLU) + Batch Normalization + Dropout (0.4).
   * **Output Layer**: 3 kelas sentimen (Softmax).

---

## 📂 Struktur Direktori

```text
├── Analisis_Sentimen_Ulasan_Shopee_di_Google_Play_Store_Menggunakan_LSTM_*.ipynb  # Notebook Colab / Training
├── app.py                     # Source code aplikasi web Streamlit
├── requirements.txt           # Daftar dependensi package Python
├── best_model_lexicon.pth     # Model bobot LSTM terlatih (Lexicon labels)
├── embedding_matrix.npy       # Matriks bobot embedding Word2Vec
├── word2vec.model             # File model Word2Vec asli dari training
├── tokenizer.pkl              # File mapping vocabulary token (word-to-index)
├── dataset_preprocessed.csv   # Dataset hasil preprocessing ulasan
├── ulasan_apk_Shoope_PlayStorev1.csv  # Dataset ulasan Shopee mentah
├── slangWords.json            # Kamus normalisasi slang/singkatan
├── eval_results.json          # Hasil metrik evaluasi model
├── *.png                      # Gambar grafik hasil visualisasi EDA & training
└── README.md                  # Dokumentasi project ini
```

---

## ⚙️ Cara Menjalankan Secara Lokal

### Prasyarat
* Python 3.10 atau versi di atasnya.
* Git terpasang di komputer Anda.

### Langkah-langkah
1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/Edong098/Analisis-Sentimen-Ulasan-Shopee-di-Google-Play-Store-Menggunakan-LSTM.git
   cd Analisis-Sentimen-Ulasan-Shopee-di-Google-Play-Store-Menggunakan-LSTM
   ```

2. **Buat virtual environment (opsional namun disarankan)**:
   ```bash
   python -m venv venv
   # Aktifkan venv
   # Di Windows:
   venv\Scripts\activate
   # Di Mac/Linux:
   source venv/bin/activate
   ```

3. **Instal dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka di browser Anda pada alamat `http://localhost:8501`.

---

## ☁️ Deployment ke Streamlit Cloud

Aplikasi ini siap dideploy ke **Streamlit Community Cloud** secara instan:
1. Hubungkan akun GitHub Anda ke [Streamlit Share](https://share.streamlit.io/).
2. Klik **New app** dan pilih repositori ini.
3. Tentukan Main file path sebagai `app.py`.
4. Klik **Deploy**. Streamlit Cloud akan membaca [requirements.txt](file:///d:/SEMESTER%206/NLP/Analisis_Sentimen_Ulasan_Shopee_di_Google_Play_Store_Menggunakan_LSTM_2301010019_2301010030_2301010036_2301010044/requirements.txt) dan mempersiapkan web server Anda dalam beberapa menit.

---

## 👥 Kelompok UAS - NLP 2026

* **NIM Ketua** - Akuisisi Dataset, Integrasi, Streamlit Web & Dokumentasi
* **NIM Anggota 2** - Text Preprocessing
* **NIM Anggota 3** - Exploratory Data Analysis (EDA)
* **NIM Anggota 4** - Feature Extraction & Tokenization
* **NIM Anggota 5** - Model LSTM & Training
