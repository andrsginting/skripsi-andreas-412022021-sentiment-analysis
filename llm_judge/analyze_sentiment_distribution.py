#llm_judge/analyze_sentiment_distribution.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np
import sys

# ======================================================
# FIX PROJECT ROOT
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

COMMENT_LABELS_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_results"

# Color mapping for sentiments
COLOR_MAP = {
    "positif": "#2ecc71",    # Green
    "netral": "#95a5a6",     # Gray
    "negatif": "#e74c3c"     # Red
}


def analyze_sentiment_distribution():
    """Menganalisis distribusi sentiment dari semua file comment labels."""
    
    if not COMMENT_LABELS_DIR.exists():
        print(f"[ERROR] Direktori {COMMENT_LABELS_DIR} tidak ditemukan!")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Collect all llm_result (comment labels)
    all_labels = []
    per_video_data = {}
    csv_files = sorted(list(COMMENT_LABELS_DIR.glob("dataset_video_*_llm.csv")))
    
    print(f"[INFO] Ditemukan {len(csv_files)} file comment labels")
    
    if not csv_files:
        print("[ERROR] Tidak ada file comment labels ditemukan!")
        return
    
    # Read all CSV files
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if "llm_result" in df.columns:
                labels = df["llm_result"].tolist()
                all_labels.extend(labels)
                
                # Extract video number from filename
                video_name = csv_file.stem.replace("dataset_video_", "").replace("_llm", "")
                per_video_data[video_name] = labels
                
                print(f"✓ {csv_file.name}: {len(labels)} comments")
            else:
                print(f"[SKIP] Kolom 'llm_result' tidak ditemukan di {csv_file.name}")
        except Exception as e:
            print(f"[ERROR] Gagal membaca {csv_file.name}: {e}")
    
    if not all_labels:
        print("[ERROR] Tidak ada data yang berhasil dikumpulkan!")
        return
    
    # ======================================================
    # STATISTIK DASAR GLOBAL
    # ======================================================
    total_comments = len(all_labels)
    label_counts = Counter(all_labels)
    
    print("\n" + "="*60)
    print("📊 ANALISIS DISTRIBUSI SENTIMENT COMMENT-LEVEL")
    print("="*60)
    print(f"\nTotal Comments Analyzed: {total_comments}")
    print(f"Unique Sentiment Labels: {len(label_counts)}")
    
    # Calculate percentages for global
    sentiment_stats = {}
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_comments) * 100
        sentiment_stats[label] = {
            "count": count,
            "percentage": percentage
        }
        print(f"\n{label.upper()}:")
        print(f"  Count: {count}")
        print(f"  Percentage: {percentage:.2f}%")
    
    # ======================================================
    # VISUALISASI 1: DISTRIBUSI PER VIDEO (STACKED BAR CHART)
    # ======================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    video_names = sorted(per_video_data.keys())
    sentiments_order = sorted(sentiment_stats.keys())
    
    # Prepare data for stacked bar chart
    sentiments_data = {sentiment: [] for sentiment in sentiments_order}
    sentiments_counts = {sentiment: [] for sentiment in sentiments_order}
    
    for video_name in video_names:
        video_labels = per_video_data[video_name]
        video_counts = Counter(video_labels)
        for sentiment in sentiments_order:
            count = video_counts.get(sentiment, 0)
            percentage = (count / len(video_labels)) * 100 if video_labels else 0
            sentiments_data[sentiment].append(percentage)
            sentiments_counts[sentiment].append(count)
    
    # Create stacked bar chart with proper color mapping
    x_pos = np.arange(len(video_names))
    width = 0.6
    bottom = np.zeros(len(video_names))
    
    for sentiment in sentiments_order:
        color = COLOR_MAP.get(sentiment, "#95a5a6")
        ax.bar(x_pos, sentiments_data[sentiment], width, bottom=bottom, 
               label=sentiment.capitalize(), color=color, edgecolor="black", linewidth=1)
        
        # Add annotations (percentage + count)
        for i in range(len(video_names)):
            if sentiments_data[sentiment][i] > 0:
                y_pos = bottom[i] + sentiments_data[sentiment][i] / 2
                ax.text(i, y_pos, 
                       f"{sentiments_data[sentiment][i]:.1f}%\n({sentiments_counts[sentiment][i]})",
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        bottom += sentiments_data[sentiment]
    
    ax.set_xlabel("Video", fontsize=12, fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    ax.set_title("Distribusi Sentimen per Video\nMain Comment Level - Analyzed by LLM", 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Video {name}" for name in video_names], rotation=45, ha='right')
    ax.legend(title="Sentiment", fontsize=10, title_fontsize=11, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_per_video.png", dpi=300)
    plt.close()
    print(f"\n✓ Gambar per video stacked bar chart disimpan: sentiment_distribution_per_video.png")
    
    # ======================================================
    # VISUALISASI 2: BAR CHART GLOBAL
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels_sorted = sorted(sentiment_stats.keys())
    counts = [sentiment_stats[label]["count"] for label in labels_sorted]
    colors = [COLOR_MAP.get(label, "#95a5a6") for label in labels_sorted]
    
    bars = ax.bar(labels_sorted, counts, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlabel("Sentiment", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title("Sentiment Distribution (All Videos)\nMain Comment Level - Analyzed by LLM", 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_count.png", dpi=300)
    plt.close()
    print(f"✓ Gambar bar chart global disimpan: sentiment_distribution_count.png")
    
    # ======================================================
    # VISUALISASI 3: PIE CHART GLOBAL
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    counts_pie = [sentiment_stats[label]["count"] for label in labels_sorted]
    colors_pie = [COLOR_MAP.get(label, "#95a5a6") for label in labels_sorted]
    
    # Create labels with count information
    pie_labels = [
        f"{label.capitalize()}\n{sentiment_stats[label]['count']} comments\n({sentiment_stats[label]['percentage']:.1f}%)"
        for label in labels_sorted
    ]
    
    wedges, texts, autotexts = ax.pie(
        counts_pie,
        labels=pie_labels,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax.set_title("Sentiment Distribution (All Videos - Percentage)\nMain Comment Level - Analyzed by LLM", 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_pie.png", dpi=300)
    plt.close()
    print(f"✓ Gambar pie chart global disimpan: sentiment_distribution_pie.png")
    
    # ======================================================
    # VISUALISASI 4: HORIZONTAL BAR CHART GLOBAL
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_labels = list(sentiment_stats.keys())
    bar_counts = [sentiment_stats[label]["count"] for label in bar_labels]
    bar_percentages = [sentiment_stats[label]["percentage"] for label in bar_labels]
    colors_hbar = [COLOR_MAP.get(label, "#95a5a6") for label in bar_labels]
    
    bars = ax.barh(bar_labels, bar_counts, color=colors_hbar, edgecolor="black", linewidth=1.5)
    
    # Add labels
    for i, (bar, count, pct) in enumerate(zip(bars, bar_counts, bar_percentages)):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height()/2.,
            f'{count} ({pct:.1f}%)',
            ha='left',
            va='center',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1)
        )
    
    ax.set_xlabel("Count", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sentiment", fontsize=12, fontweight='bold')
    ax.set_title("Sentiment Distribution (All Videos)\nMain Comment Level - Analyzed by LLM", 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_horizontal.png", dpi=300)
    plt.close()
    print(f"✓ Gambar horizontal bar chart global disimpan: sentiment_distribution_horizontal.png")
    
    # ======================================================
    # SAVE STATISTIK KE CSV
    # ======================================================
    stats_df = pd.DataFrame({
        "Sentiment": list(sentiment_stats.keys()),
        "Count": [sentiment_stats[label]["count"] for label in sentiment_stats.keys()],
        "Percentage": [f"{sentiment_stats[label]['percentage']:.2f}%" for label in sentiment_stats.keys()]
    })
    
    stats_df.to_csv(OUTPUT_DIR / "sentiment_distribution_stats.csv", index=False)
    print(f"\n✓ Statistik global disimpan: sentiment_distribution_stats.csv")
    
    # ======================================================
    # SAVE SUMMARY REPORT
    # ======================================================
    summary_text = f"""
COMMENT-LEVEL SENTIMENT ANALYSIS REPORT
{'='*60}

Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Files Analyzed: {len(csv_files)}
Total Comments Analyzed: {total_comments}

SENTIMENT DISTRIBUTION (GLOBAL):
{'-'*60}
"""
    
    for label in sorted(sentiment_stats.keys()):
        count = sentiment_stats[label]["count"]
        percentage = sentiment_stats[label]["percentage"]
        summary_text += f"\n{label.upper()}:\n"
        summary_text += f"  Count: {count}\n"
        summary_text += f"  Percentage: {percentage:.2f}%\n"
    
    summary_text += f"\n{'='*60}\n\nDISTRIBUSI PER VIDEO:\n{'-'*60}\n"
    
    for video_name in video_names:
        video_labels = per_video_data[video_name]
        video_counts = Counter(video_labels)
        summary_text += f"\nVideo {video_name}:\n"
        summary_text += f"  Total Comments: {len(video_labels)}\n"
        for sentiment in sorted(video_counts.keys()):
            count = video_counts[sentiment]
            pct = (count / len(video_labels)) * 100
            summary_text += f"    {sentiment.capitalize()}: {count} ({pct:.2f}%)\n"
    
    summary_text += f"\n{'='*60}\n"
    
    with open(OUTPUT_DIR / "sentiment_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"✓ Laporan ringkas disimpan: sentiment_analysis_report.txt")
    
    print("\n" + "="*60)
    print("🎉 Analisis sentiment distribution comment-level selesai!")
    print("="*60)
    print(f"\nHasil analisis disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze_sentiment_distribution()
