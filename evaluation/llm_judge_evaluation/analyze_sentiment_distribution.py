# evaluation/llm_judge_evaluation/analyze_sentiment_distribution.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sys

# ======================================================
# FIX PROJECT ROOT
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

BASE_RESULT = PROJECT_ROOT / "evaluation/llm_judge_evaluation/results"
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_results"


def analyze_sentiment_distribution():
    """Menganalisis distribusi sentiment comment-level dari semua experiment."""
    
    if not BASE_RESULT.exists():
        print(f"[ERROR] Direktori {BASE_RESULT} tidak ditemukan!")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    experiments = [d for d in BASE_RESULT.iterdir() if d.is_dir()]
    
    print(f"[INFO] Ditemukan {len(experiments)} experiments")
    
    if not experiments:
        print("[ERROR] Tidak ada experiment ditemukan!")
        return
    
    # ======================================================
    # COLLECT DATA PER EXPERIMENT
    # ======================================================
    experiment_data = {}
    all_labels_global = []
    
    for exp_dir in sorted(experiments):
        exp_name = exp_dir.name
        merged_dir = exp_dir / "merged"
        
        if not merged_dir.exists():
            print(f"[SKIP] Direktori merged tidak ditemukan di {exp_name}")
            continue
        
        exp_labels = []
        merged_files = list(merged_dir.glob("*_merged.csv"))
        
        if not merged_files:
            print(f"[SKIP] Tidak ada merged file di {exp_name}")
            continue
        
        # Read all merged files in this experiment
        for csv_file in merged_files:
            try:
                df = pd.read_csv(csv_file)
                if "llm_label" in df.columns:
                    labels = df["llm_label"].tolist()
                    exp_labels.extend(labels)
                    all_labels_global.extend(labels)
            except Exception as e:
                print(f"[ERROR] Gagal membaca {csv_file.name}: {e}")
        
        if exp_labels:
            experiment_data[exp_name] = exp_labels
            print(f"✓ {exp_name}: {len(exp_labels)} comments")
    
    if not all_labels_global:
        print("[ERROR] Tidak ada data yang berhasil dikumpulkan!")
        return
    
    # ======================================================
    # STATISTIK GLOBAL
    # ======================================================
    total_comments = len(all_labels_global)
    label_counts_global = Counter(all_labels_global)
    
    print("\n" + "="*60)
    print("📊 ANALISIS DISTRIBUSI SENTIMENT COMMENT-LEVEL (KESELURUHAN)")
    print("="*60)
    print(f"\nTotal Experiments: {len(experiment_data)}")
    print(f"Total Comments Analyzed: {total_comments}")
    print(f"Unique Sentiment Labels: {len(label_counts_global)}")
    
    # Calculate percentages for global
    sentiment_stats_global = {}
    for label, count in sorted(label_counts_global.items()):
        percentage = (count / total_comments) * 100
        sentiment_stats_global[label] = {
            "count": count,
            "percentage": percentage
        }
        print(f"\n{label.upper()}:")
        print(f"  Count: {count}")
        print(f"  Percentage: {percentage:.2f}%")
    
    # ======================================================
    # VISUALISASI 1: BAR CHART (GLOBAL)
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels_sorted = sorted(sentiment_stats_global.keys())
    counts = [sentiment_stats_global[label]["count"] for label in labels_sorted]
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]  # green, gray, red
    
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
    ax.set_title("Comment-Level Sentiment Distribution (All Videos - Keseluruhan)\nMain Comment Level - Analyzed by LLM", 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_count_global.png", dpi=300)
    plt.close()
    print(f"\n✓ Gambar bar chart disimpan: sentiment_distribution_count_global.png")
    
    # ======================================================
    # VISUALISASI 2: PIE CHART (GLOBAL)
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    counts_pie = [sentiment_stats_global[label]["count"] for label in labels_sorted]
    
    wedges, texts, autotexts = ax.pie(
        counts_pie,
        labels=labels_sorted,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title("Comment-Level Sentiment Distribution (All Videos - Keseluruhan)\nMain Comment Level - Analyzed by LLM", 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_pie_global.png", dpi=300)
    plt.close()
    print(f"✓ Gambar pie chart disimpan: sentiment_distribution_pie_global.png")
    
    # ======================================================
    # VISUALISASI 3: HORIZONTAL BAR CHART (GLOBAL)
    # ======================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_labels = list(sentiment_stats_global.keys())
    bar_counts = [sentiment_stats_global[label]["count"] for label in bar_labels]
    bar_percentages = [sentiment_stats_global[label]["percentage"] for label in bar_labels]
    
    bars = ax.barh(bar_labels, bar_counts, color=colors, edgecolor="black", linewidth=1.5)
    
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
    ax.set_title("Comment-Level Sentiment Distribution (All Videos - Keseluruhan)\nMain Comment Level - Analyzed by LLM", 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution_horizontal_global.png", dpi=300)
    plt.close()
    print(f"✓ Gambar horizontal bar chart disimpan: sentiment_distribution_horizontal_global.png")
    
    # ======================================================
    # VISUALISASI PER EXPERIMENT: COMPARISON
    # ======================================================
    if len(experiment_data) > 1:
        exp_names_list = sorted(experiment_data.keys())
        
        # Prepare data for comparison
        comparison_data = {}
        for label in sorted(sentiment_stats_global.keys()):
            comparison_data[label] = []
            for exp_name in exp_names_list:
                exp_labels = experiment_data[exp_name]
                exp_label_counts = Counter(exp_labels)
                count = exp_label_counts.get(label, 0)
                percentage = (count / len(exp_labels)) * 100 if exp_labels else 0
                comparison_data[label].append(percentage)
        
        # Create comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(exp_names_list))
        width = 0.25
        
        for i, (label, percentages) in enumerate(comparison_data.items()):
            offset = (i - 1) * width
            ax.bar([xi + offset for xi in x], percentages, width, label=label, color=colors[i], edgecolor="black", linewidth=1)
        
        ax.set_xlabel("Experiment", fontsize=12, fontweight='bold')
        ax.set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
        ax.set_title("Comment-Level Sentiment Distribution Per Experiment (Comparison)\nMain Comment Level - Analyzed by LLM", 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names_list, rotation=45, ha='right')
        ax.legend(title="Sentiment", fontsize=10, title_fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sentiment_distribution_comparison_per_experiment.png", dpi=300)
        plt.close()
        print(f"✓ Gambar comparison chart disimpan: sentiment_distribution_comparison_per_experiment.png")
    
    # ======================================================
    # SAVE STATISTIK KE CSV
    # ======================================================
    stats_df = pd.DataFrame({
        "Sentiment": list(sentiment_stats_global.keys()),
        "Count": [sentiment_stats_global[label]["count"] for label in sentiment_stats_global.keys()],
        "Percentage": [f"{sentiment_stats_global[label]['percentage']:.2f}%" for label in sentiment_stats_global.keys()]
    })
    
    stats_df.to_csv(OUTPUT_DIR / "sentiment_distribution_stats_global.csv", index=False)
    print(f"\n✓ Statistik global disimpan: sentiment_distribution_stats_global.csv")
    
    # ======================================================
    # SAVE SUMMARY REPORT
    # ======================================================
    summary_text = f"""
COMMENT-LEVEL SENTIMENT ANALYSIS REPORT (KESELURUHAN)
{'='*70}

Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Experiments Analyzed: {len(experiment_data)}
Total Comments Analyzed: {total_comments}

SENTIMENT DISTRIBUTION (GLOBAL):
{'-'*70}
"""
    
    for label in sorted(sentiment_stats_global.keys()):
        count = sentiment_stats_global[label]["count"]
        percentage = sentiment_stats_global[label]["percentage"]
        summary_text += f"\n{label.upper()}:\n"
        summary_text += f"  Count: {count}\n"
        summary_text += f"  Percentage: {percentage:.2f}%\n"
    
    summary_text += f"\n{'='*70}\n\nEXPERIMENT BREAKDOWN:\n{'-'*70}\n"
    
    for exp_name in sorted(experiment_data.keys()):
        exp_labels = experiment_data[exp_name]
        exp_counts = Counter(exp_labels)
        summary_text += f"\n{exp_name}:\n"
        summary_text += f"  Total Comments: {len(exp_labels)}\n"
        for label in sorted(exp_counts.keys()):
            count = exp_counts[label]
            pct = (count / len(exp_labels)) * 100
            summary_text += f"    {label}: {count} ({pct:.2f}%)\n"
    
    summary_text += f"\n{'='*70}\n"
    
    with open(OUTPUT_DIR / "sentiment_analysis_report_global.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"✓ Laporan ringkas disimpan: sentiment_analysis_report_global.txt")
    
    print("\n" + "="*60)
    print("🎉 Analisis sentiment distribution comment-level selesai!")
    print("="*60)
    print(f"\nHasil analisis disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze_sentiment_distribution()
