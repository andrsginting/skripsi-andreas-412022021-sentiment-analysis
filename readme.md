# TUGAS AKHIR
## ANALISIS SENTIMEN PUBLIK TERHADAP PAPARAN MENTERI KEUANGAN RI PERIODE 2025–2029 PADA KOMENTAR YOUTUBE MENGGUNAKAN INDOBERT DAN PEMBOBOTAN LIKES

**Nama:** Andreas Nicolas Ginting  
**UNIV:** UKRIDA
**PRODI:** INFORMATIKA
**NIM:** 412022021  

---

## 📘 Deskripsi Proyek
Repository ini berisi implementasi lengkap pipeline **analisis sentimen komentar YouTube** untuk mendukung penelitian tugas akhir.  
Pendekatan utama yang digunakan adalah:

- **IndoBERT** sebagai model analisis sentimen berbasis Transformer
- **Pembobotan likes** untuk merepresentasikan pengaruh opini
- **Agregasi sentimen berbasis thread diskusi**
- **Large Language Model (GPT-4o-mini) sebagai ground truth (LLM as a Judge)**

Pipeline dirancang **end-to-end**, mulai dari scraping data mentah hingga evaluasi model secara kuantitatif dan visual.

---

## 📁 Struktur Direktori Utama
```

├── scrapping/                     # Scraping komentar YouTube
├── cleaning/                      # Pembersihan dan preprocessing data
├── sentiment/                     # Inferensi sentimen IndoBERT
├── weighted_average_summary/      # Ringkasan sentimen level video
├── llm_judge/                     # Ground truth comment-level (LLM)
├── new_llm_judge/                 # Ground truth thread-level (LLM)
├── evaluation/                    # Evaluasi IndoBERT vs LLM (comment-level)
├── new_evaluation/                # Evaluasi IndoBERT vs LLM (thread-level)
├── run_scraper.py
├── run_comment_statistics.py
├── run_cleaning.py
├── run_count_words.py
└── requirements.txt

````

---

## ⚙️ Persiapan Awal

### 1️⃣ Clone Repository
```bash
git clone https://github.com/andrsginting/skripsi-andreas-412022021-sentiment-analysis.git
cd skripsi-andreas-412022021-sentiment-analysis
````

### 2️⃣ (Opsional) Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependensi

```bash
pip install -r requirements.txt
```

### 4️⃣ Konfigurasi API Key (WAJIB untuk LLM Judge)

Buat file `.env` di root project:

```
OPENAI_API_KEY=YOUR_API_KEY
```

---

## 🚀 Alur Eksekusi Pipeline (STEP-BY-STEP DETAIL)

---

## STEP 1 — Scraping Komentar YouTube

**File:** `run_scraper.py`

### Proses:

* Mengambil komentar utama dan balasan (reply) dari maksimal **6 video YouTube**
* Setiap komentar dikaitkan dengan `thread_id`
* Menyimpan jumlah likes dan status reply
* Mendukung mode **browser terlihat** atau **headless**

```bash
python run_scraper.py
```

### Output:

* Folder: `scrapping/dataset/`
* File: `dataset_video_X.csv`
* Kolom utama:

  * `thread_id`
  * `comment`
  * `likes_count`
  * `is_reply`

---

## STEP 2 — Statistik Dasar Komentar

**File:** `run_comment_statistics.py`

### Proses:

* Menghitung jumlah komentar total
* Memisahkan komentar utama dan reply
* Menghitung total dan rata-rata likes
* Mengidentifikasi komentar dengan likes tertinggi

```bash
python run_comment_statistics.py
```

### Output:

* `statistics/comment_scraping_stats/comment_scraping_summary.csv`
* File detail per video

---

## STEP 3 — Cleaning & Preprocessing Data

**File:** `run_cleaning.py`

### Proses:

* Lowercasing
* Menghapus URL, emoji, simbol
* Normalisasi teks
* Menghasilkan kolom `cleaned_comment`

```bash
python run_cleaning.py
```

### Output:

* Folder: `cleaning/dataset/`
* File: `*_cleaned.csv`

---

## STEP 4 — Analisis Panjang Komentar

**File:** `run_count_words.py`

### Proses:

* Menghitung jumlah kata per komentar
* Menghasilkan statistik rata-rata panjang komentar

```bash
python run_count_words.py
```

### Output:

* Folder: `dataset_count_word/`
* File ringkasan global & per video

---

## STEP 5 — Inferensi Sentimen Menggunakan IndoBERT

**File:** `sentiment/runners/run_sentiment_inference.py`

### Proses:

* Melakukan klasifikasi sentimen pada setiap komentar
* Model: `mdhugol/indonesia-bert-sentiment-classification`
* Label: `positive`, `neutral`, `negative`

```bash
python sentiment/runners/run_sentiment_inference.py
```

### Output:

* Folder: `sentiment/dataset/sentiment/`
* File: `*_sentiment.csv`
* Kolom: `predicted_label`, `confidence_score`

---

## STEP 6 — Contextual Sentiment Adjustment

**File:** `sentiment/runners/run_contextual_adjustment.py`

### Proses:

* Menyesuaikan sentimen komentar berdasarkan konteks reply dan main comment
* Eksperimen bobot:

  * 60% main – 40% reply
  * 70% main – 30% reply
  * 80% main – 20% reply

```bash
python sentiment/runners/run_contextual_adjustment.py
```

### Output:

* Folder: `sentiment/dataset/contextual/{experiment}/`
* File: `*_contextual.csv`

---

## STEP 7 — Agregasi Sentimen Level Thread

**File:** `sentiment/runners/run_thread_aggregation.py`

### Proses:

* Menggabungkan sentimen komentar dalam satu thread
* Menghasilkan skor sentimen berbobot likes

```bash
python sentiment/runners/run_thread_aggregation.py
```

### Output:

* Folder: `sentiment/dataset/summary/{experiment}/`
* File: `*_cleaned_summary.csv`

---

## STEP 8 — Ringkasan Sentimen Level Video

**File:** `weighted_average_summary/video_sentiment_summary.py`

### Proses:

* Menghitung proporsi sentimen positif, netral, negatif per video
* Berdasarkan agregasi thread

```bash
python weighted_average_summary/video_sentiment_summary.py
```

### Output:

* `video_sentiment_overview.csv`

---

## STEP 9 — Visualisasi Sentimen

**File:** `weighted_average_summary/generate_sentiment_charts.py`

### Proses:

* Membuat bar chart per video
* Membuat pie chart keseluruhan

```bash
python weighted_average_summary/generate_sentiment_charts.py
```

### Output:

* File PNG (bar chart & pie chart)

---

## 🤖 Ground Truth Menggunakan LLM (GPT-4o-mini)

## STEP 10 — LLM Judge (Comment-Level)

**File:** `llm_judge/runners/run_llm_judge.py`

### Proses:

* Memberi label sentimen komentar sebagai ground truth
* Menggunakan prompt terkontrol

```bash
python llm_judge/runners/run_llm_judge.py
```

### Output:

* `llm_judge/output/*_llm.csv`

---

## STEP 11 — Analisis Distribusi Label LLM

```bash
python llm_judge/analyze_sentiment_distribution.py
```

### Output:

* Statistik dan visualisasi distribusi sentimen comment-level

---

## STEP 12 — Ground Truth Thread-Level (LLM)

```bash
python new_llm_judge/thread_evaluation/builders/build_thread_json.py
python new_llm_judge/thread_evaluation/runners/run_thread_judge.py
python new_llm_judge/thread_evaluation/runners/run_distribution_labels_thread.py
```

### Output:

* Label sentimen thread-level
* Visualisasi distribusi thread

---

## 📊 Evaluasi Model IndoBERT

## STEP 13 — Evaluasi Comment-Level

```bash
python evaluation/run_evaluate_indobert_vs_llm.py
```

### Output:

* Accuracy, Precision, Recall, F1-score
* Classification report per video & keseluruhan

---

## STEP 14 — Confusion Matrix Comment-Level

```bash
python evaluation/generate_confusion_matrix.py
```

### Output:

* Confusion matrix (count & normalized)
* File PNG dan TXT

---

## STEP 15 — Evaluasi Thread-Level (Final)

```bash
python new_evaluation/thread_evaluation/runners/run_thread_evaluation.py
python new_evaluation/thread_evaluation/runners/run_confusion_matrix_per_video.py
```

### Output:

* Evaluasi performa IndoBERT pada level thread
* Confusion matrix per eksperimen

---

## 🧠 Catatan Akademik

Pipeline ini dirancang untuk:

* Mendukung validitas penelitian skripsi
* Memungkinkan replikasi eksperimen
* Membandingkan model ML dengan LLM sebagai ground truth

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik dan penelitian tugas akhir.


