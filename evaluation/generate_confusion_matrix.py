# evaluation/generate_confusion_matrix.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# ======================================================
# PATH
# ======================================================
INDOBERT_DIR = "sentiment/dataset/sentiment/"
LLM_DIR = "llm_judge/output/"
OUT_DIR = "evaluation/results/"

os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP = {
    "positive": "positif",
    "negative": "negatif",
    "neutral": "netral",
    "positif": "positif",
    "negatif": "negatif",
    "netral": "netral"
}

LABEL_ORDER = ["positif", "netral", "negatif"]


def normalize(label):
    """Normalize label ke format standar"""
    return LABEL_MAP.get(str(label).strip().lower(), None)


def extract_video_id(filename):
    """Extract video ID dari filename"""
    return filename.split("_cleaned")[0]


def generate_confusion_matrix_heatmap(y_true, y_pred, title, save_path):
    """Generate dan save confusion matrix heatmap"""
    
    # Create confusion matrix
    cm = confusion_matrix(
        y_true, y_pred, 
        labels=LABEL_ORDER,
        normalize=None
    )
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        cbar_kws={'label': 'Frekuensi'},
        linewidths=2,
        linecolor='black',
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    
    plt.xlabel('Prediksi Model IndoBERT', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth (GPT-4)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()
    
    return cm


def generate_normalized_heatmap(y_true, y_pred, title, save_path):
    """Generate normalized confusion matrix (percentase per row)"""
    
    # Create confusion matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=LABEL_ORDER,
        normalize=None
    )
    
    # Normalize per row (setiap label groundtruth)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='YlOrRd',
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        cbar_kws={'label': 'Persentase (%)'},
        linewidths=2,
        linecolor='black',
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    plt.xlabel('Prediksi Model IndoBERT', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth (GPT-4)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Normalized confusion matrix saved: {save_path}")
    plt.close()


def print_confusion_matrix_stats(cm, label):
    """Print detailed confusion matrix statistics"""
    print(f"\n{'='*70}")
    print(f"CONFUSION MATRIX - {label}")
    print(f"{'='*70}")
    
    print("\nMatrix shape: 3×3 (Positif, Netral, Negatif)")
    print("\nRaw Count:")
    print(f"{'':15} | {'Pred Positif':>12} | {'Pred Netral':>12} | {'Pred Negatif':>12}")
    print("-" * 70)
    
    for i, true_label in enumerate(LABEL_ORDER):
        print(f"True {true_label:9} | {cm[i,0]:>12} | {cm[i,1]:>12} | {cm[i,2]:>12}")
    
    print("\n" + "-" * 70)
    
    # Calculate metrics per class
    print("\nMetrics per Class:")
    print(f"{'Class':15} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precision':>10} | {'Recall':>10}")
    print("-" * 70)
    
    for i, label_name in enumerate(LABEL_ORDER):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{label_name:15} | {tp:>6} | {fp:>6} | {fn:>6} | {precision:>10.4f} | {recall:>10.4f}")
    
    print("=" * 70 + "\n")


def evaluate_pair(video_id, indo_path, llm_path):
    """Evaluate satu video pair"""
    
    df_indo = pd.read_csv(indo_path)
    df_llm = pd.read_csv(llm_path)
    
    # Merge berdasarkan kolom umum
    df = df_indo.merge(
        df_llm,
        on=["thread_id", "cleaned_comment", "likes_count", "is_reply"],
        how="inner"
    )
    
    if df.empty:
        print(f"[WARN] Tidak ada data cocok untuk {video_id}")
        return None
    
    y_true = df["predicted_label"].apply(normalize)
    y_pred = df["llm_result"].apply(normalize)
    
    # Filter out null values
    valid_mask = y_true.notnull() & y_pred.notnull()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        print(f"[WARN] Tidak ada data valid setelah normalisasi untuk {video_id}")
        return None
    
    return y_true.values, y_pred.values


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("CONFUSION MATRIX GENERATOR - IndoBERT vs LLM Judge")
    print("="*70 + "\n")
    
    # Store all predictions
    all_true, all_pred = [], []
    per_video_results = []
    
    indo_files = sorted(glob.glob(os.path.join(INDOBERT_DIR, "*_sentiment.csv")))
    
    for indo_path in indo_files:
        filename = os.path.basename(indo_path)
        video_id = extract_video_id(filename)
        
        # Find corresponding LLM file
        llm_candidates = glob.glob(
            os.path.join(LLM_DIR, f"{video_id}*_llm.csv")
        )
        
        if not llm_candidates:
            print(f"[SKIP] LLM file tidak ditemukan untuk {video_id}")
            continue
        
        llm_path = llm_candidates[0]
        
        print(f"📊 Processing {video_id}...")
        
        result = evaluate_pair(video_id, indo_path, llm_path)
        
        if result is not None:
            y_true, y_pred = result
            
            # Extend all predictions
            all_true.extend(y_true)
            all_pred.extend(y_pred)
            
            # Generate per-video confusion matrix
            cm = confusion_matrix(
                y_true, y_pred,
                labels=LABEL_ORDER,
                normalize=None
            )
            
            # Save per-video heatmap
            save_path_count = os.path.join(
                OUT_DIR, 
                f"{video_id}_confusion_matrix.png"
            )
            save_path_norm = os.path.join(
                OUT_DIR,
                f"{video_id}_confusion_matrix_normalized.png"
            )
            
            generate_confusion_matrix_heatmap(
                y_true, y_pred,
                f"Confusion Matrix - {video_id}\n(Count)",
                save_path_count
            )
            
            generate_normalized_heatmap(
                y_true, y_pred,
                f"Confusion Matrix - {video_id}\n(Normalized)",
                save_path_norm
            )
            
            # Print stats
            print_confusion_matrix_stats(cm, video_id)
            
            # Store results
            per_video_results.append({
                'video': video_id,
                'total_samples': len(y_true),
                'confusion_matrix': cm.tolist()
            })
    
    # ===============================
    # OVERALL CONFUSION MATRIX
    # ===============================
    if all_true and all_pred:
        print("\n" + "="*70)
        print("GENERATING OVERALL CONFUSION MATRIX")
        print("="*70 + "\n")
        
        # Generate overall heatmaps
        save_path_overall = os.path.join(
            OUT_DIR,
            "overall_confusion_matrix.png"
        )
        save_path_overall_norm = os.path.join(
            OUT_DIR,
            "overall_confusion_matrix_normalized.png"
        )
        
        generate_confusion_matrix_heatmap(
            all_true, all_pred,
            "Confusion Matrix - OVERALL (All Videos)\n(Count)",
            save_path_overall
        )
        
        generate_normalized_heatmap(
            all_true, all_pred,
            "Confusion Matrix - OVERALL (All Videos)\n(Normalized)",
            save_path_overall_norm
        )
        
        # Get overall CM and print stats
        cm_overall = confusion_matrix(
            all_true, all_pred,
            labels=LABEL_ORDER,
            normalize=None
        )
        
        print_confusion_matrix_stats(cm_overall, "OVERALL")
        
        # Save summary to file
        summary_file = os.path.join(OUT_DIR, "confusion_matrix_summary.txt")
        with open(summary_file, "w") as f:
            f.write("="*70 + "\n")
            f.write("CONFUSION MATRIX SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Samples: {len(all_true)}\n")
            f.write(f"Label Order: {LABEL_ORDER}\n\n")
            
            f.write("OVERALL CONFUSION MATRIX (Count):\n")
            f.write(str(cm_overall) + "\n\n")
            
            f.write("OVERALL CONFUSION MATRIX (Normalized %):\n")
            cm_norm = cm_overall.astype('float') / cm_overall.sum(axis=1)[:, np.newaxis]
            f.write(np.array2string(cm_norm * 100, precision=2) + "\n")
        
        print(f"\n✓ Summary saved: {summary_file}")
    
    print("\n" + "="*70)
    print("✓ SELESAI! Semua confusion matrix telah dibuat")
    print("="*70)
    print(f"\n📁 Output tersimpan di: {OUT_DIR}")
    print("\nFile yang dihasilkan:")
    print("  - [video_id]_confusion_matrix.png")
    print("  - [video_id]_confusion_matrix_normalized.png")
    print("  - overall_confusion_matrix.png")
    print("  - overall_confusion_matrix_normalized.png")
    print("  - confusion_matrix_summary.txt\n")


if __name__ == "__main__":
    main()
