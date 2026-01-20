
# Sentiment Analysis YouTube Comments using IndoBERT and LLM as a Judge

Repository ini berisi implementasi lengkap pipeline **scraping komentar YouTube**, **analisis sentimen berbasis IndoBERT**, serta **evaluasi model menggunakan pendekatan LLM as a Judge (GPT-4o-mini)** pada level komentar dan thread.  
Proyek ini dikembangkan sebagai bagian dari penelitian skripsi.

---

## 📌 Alur Umum Pipeline

1. Scraping komentar YouTube (multi-video)
2. Statistik hasil scraping
3. Pembersihan data (cleaning)
4. Analisis panjang komentar
5. Sentiment inference menggunakan IndoBERT
6. Penyesuaian sentimen berbasis konteks (contextual adjustment)
7. Agregasi sentimen level thread
8. Ringkasan sentimen per video
9. Visualisasi distribusi sentimen
10. Anotasi sentimen menggunakan LLM (GPT-4o-mini)
11. Evaluasi IndoBERT vs LLM (comment-level & thread-level)
12. Confusion matrix dan analisis error

---

## 🧰 Persiapan Awal

### 1️⃣ Clone Repository
```bash
git clone https://github.com/andrsginting/skripsi-andreas-412022021-sentiment-analysis.git
cd skripsi-andreas-412022021-sentiment-analysis
````

### 2️⃣ Install Dependencies

Disarankan menggunakan virtual environment.

```bash
pip install -r requirements.txt
```

### 3️⃣ Setup Environment Variable (LLM Judge)

Buat file `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🚀 Langkah Eksekusi Pipeline (Step-by-Step)

---

## 1. Scraping Komentar YouTube

**File:** `run_scraper.py`

**Tujuan:**
Mengambil komentar dan balasan dari hingga 6 video YouTube.

**Perintah:**

```bash
python run_scraper.py
```

**Input:**

* URL video YouTube
* Mode browser (visible / headless)

**Output:**

```
scrapping/dataset/
└── dataset_video_X.csv
```

---

## 2. Statistik Hasil Scraping

**File:** `run_comment_statistics.py`

**Tujuan:**
Menghasilkan statistik dasar komentar (jumlah komentar, likes, reply, dll).

**Perintah:**

```bash
python run_comment_statistics.py
```

**Output:**

```
statistics/comment_scraping_stats/
├── comment_scraping_summary.csv
└── dataset_video_X_details.csv
```

---

## 3. Cleaning Dataset

**File:** `run_cleaning.py`

**Tujuan:**
Membersihkan teks komentar (normalisasi, hapus noise, dll).

**Perintah:**

```bash
python run_cleaning.py
```

**Output:**

```
cleaning/dataset/
└── dataset_video_X_cleaned.csv
```

---

## 4. Analisis Jumlah Kata

**File:** `run_count_words.py`

**Tujuan:**
Menghitung panjang komentar sebagai analisis karakteristik data.

**Perintah:**

```bash
python run_count_words.py
```

**Output:**

```
dataset_count_word/
├── count_words_dataset_video_X.csv
├── summary_avg_words_per_file.csv
└── summary_global_avg_words.csv
```

---

## 5. Sentiment Inference (IndoBERT)

**File:** `sentiment/runners/run_sentiment_inference.py`

**Tujuan:**
Melakukan klasifikasi sentimen komentar menggunakan IndoBERT.

**Perintah:**

```bash
python sentiment/runners/run_sentiment_inference.py
```

**Model:**

```
mdhugol/indonesia-bert-sentiment-classification
```

**Output:**

```
sentiment/dataset/sentiment/
└── dataset_video_X_sentiment.csv
```

---

## 6. Contextual Adjustment

**File:** `sentiment/runners/run_contextual_adjustment.py`

**Tujuan:**
Menggabungkan sentimen komentar utama dan balasan dengan bobot berbeda (60/40, 70/30, 80/20).

**Perintah:**

```bash
python sentiment/runners/run_contextual_adjustment.py
```

**Output:**

```
sentiment/dataset/contextual/
├── 60_main_sentiment/
├── 70_main_sentiment/
└── 80_main_sentiment/
```

---

## 7. Agregasi Thread-Level

**File:** `sentiment/runners/run_thread_aggregation.py`

**Tujuan:**
Menghasilkan satu skor sentimen untuk setiap thread diskusi.

**Perintah:**

```bash
python sentiment/runners/run_thread_aggregation.py
```

**Output:**

```
sentiment/dataset/summary/{experiment}/
└── dataset_video_X_cleaned_summary.csv
```

---

## 8. Ringkasan Sentimen per Video

**File:** `weighted_average_summary/video_sentiment_summary.py`

**Tujuan:**
Menghasilkan distribusi sentimen per video.

**Perintah:**

```bash
python weighted_average_summary/video_sentiment_summary.py
```

**Output:**

```
weighted_average_summary/{experiment}/video_sentiment_overview.csv
```

---

## 9. Visualisasi Sentimen

**File:** `weighted_average_summary/generate_sentiment_charts.py`

**Tujuan:**
Membuat bar chart dan pie chart sentimen.

**Perintah:**

```bash
python weighted_average_summary/generate_sentiment_charts.py
```

**Output:**

```
bar_chart_sentiment_per_video.png
pie_chart_overall_sentiment.png
```

---

## 10. LLM as a Judge – Comment Level

**File:** `llm_judge/runners/run_llm_judge.py`

**Tujuan:**
Menghasilkan label sentimen ground truth menggunakan GPT-4o-mini.

**Perintah:**

```bash
python llm_judge/runners/run_llm_judge.py
```

**Output:**

```
llm_judge/output/
└── dataset_video_X_llm.csv
```

---

## 11. Evaluasi IndoBERT vs LLM (Comment Level)

**File:** `evaluation/run_evaluate_indobert_vs_llm.py`

**Tujuan:**
Menghitung Accuracy, Precision, Recall, dan F1-Score.

**Perintah:**

```bash
python evaluation/run_evaluate_indobert_vs_llm.py
```

**Output:**

```
evaluation/results/
├── summary_per_video.csv
├── overall_classification_report.txt
└── dataset_video_X_classification_report.txt
```

---

## 12. Confusion Matrix

**File:** `evaluation/generate_confusion_matrix.py`

**Tujuan:**
Visualisasi kesalahan klasifikasi IndoBERT terhadap LLM.

**Perintah:**

```bash
python evaluation/generate_confusion_matrix.py
```

**Output:**

```
evaluation/results/
├── *_confusion_matrix.png
├── *_confusion_matrix_normalized.png
└── confusion_matrix_summary.txt
```

---

## 🧠 Catatan Penting

* Pipeline **harus dijalankan berurutan**
* LLM Judge membutuhkan **API Key OpenAI**
* Waktu eksekusi LLM bergantung pada jumlah komentar

---

## 📜 Lisensi

Proyek ini dikembangkan untuk keperluan akademik (skripsi).
Penggunaan ulang diperbolehkan dengan mencantumkan sumber.


