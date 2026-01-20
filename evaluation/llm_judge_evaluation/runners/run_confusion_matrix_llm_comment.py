# evaluation/llm_judge_evaluation/runners/run_confusion_matrix_llm_comment.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

# ======================================================
# FIX PROJECT ROOT
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

LABEL_ORDER = ["positif", "netral", "negatif"]
BASE_RESULT = PROJECT_ROOT / "evaluation/llm_judge_evaluation/results"


def plot_cm(cm, title, out_path, normalized=False, experiment_name=""):
    try:
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalized else "d",
            cmap="Reds" if normalized else "Blues",
            xticklabels=LABEL_ORDER,
            yticklabels=LABEL_ORDER,
            linewidths=1,
            linecolor="black"
        )
        plt.xlabel("Prediksi IndoBERT")
        plt.ylabel("Ground Truth (GPT - Comment Level)")
        full_title = f"[{experiment_name}] {title}" if experiment_name else title
        plt.title(full_title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Gagal membuat gambar {out_path}: {e}")
        plt.close()


def main():
    print(f"[DEBUG] BASE_RESULT path: {BASE_RESULT}")
    print(f"[DEBUG] BASE_RESULT exists: {BASE_RESULT.exists()}")
    
    if not BASE_RESULT.exists():
        print("[ERROR] BASE_RESULT directory tidak ditemukan!")
        return
    
    experiments = [d for d in BASE_RESULT.iterdir() if d.is_dir()]
    if not experiments:
        print("[INFO] Tidak ada hasil evaluasi ditemukan.")
        return

    print(f"[INFO] Ditemukan {len(experiments)} experiments")

    for exp_dir in experiments:
        exp_name = exp_dir.name
        print(f"\n📊 Confusion Matrix (LLM Comment-Level): {exp_name}")

        merged_dir = exp_dir / "merged"
        per_video_dir = exp_dir / "confusion_matrix"
        all_dir = exp_dir / "confusion_matrix_all"

        # Cek apakah merged_dir ada
        if not merged_dir.exists():
            print(f"[SKIP] Direktori merged tidak ditemukan di {exp_dir}")
            continue

        per_video_dir.mkdir(exist_ok=True, parents=True)
        all_dir.mkdir(exist_ok=True, parents=True)

        all_true, all_pred = [], []

        merged_files = list(merged_dir.glob("*_merged.csv"))
        print(f"[DEBUG] Ditemukan {len(merged_files)} merged files di {merged_dir}")
        
        if not merged_files:
            print(f"[SKIP] Tidak ada merged file di {exp_name}")
            continue

        # ===============================
        # PER VIDEO
        # ===============================
        for f in merged_files:
            try:
                video_id = f.stem.replace("_merged", "")
                df = pd.read_csv(f)

                if not {"llm_label", "indo_label"}.issubset(df.columns):
                    print(f"[SKIP] Kolom tidak lengkap di {f.name}")
                    continue

                y_true = df["llm_label"]
                y_pred = df["indo_label"]

                all_true.extend(y_true.tolist())
                all_pred.extend(y_pred.tolist())

                cm_count = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
                cm_norm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize="true")

                plot_cm(
                    cm_count,
                    f"Confusion Matrix (Count)\n{video_id}",
                    per_video_dir / f"{video_id}_count.png",
                    normalized=False,
                    experiment_name=exp_name
                )

                plot_cm(
                    cm_norm,
                    f"Confusion Matrix (Normalized)\n{video_id}",
                    per_video_dir / f"{video_id}_normalized.png",
                    normalized=True,
                    experiment_name=exp_name
                )

                print(f"✓ {video_id}")
            except Exception as e:
                print(f"[ERROR] Gagal memproses {f.name}: {e}")
                continue

        # ===============================
        # ALL VIDEOS (GLOBAL)
        # ===============================
        if all_true and all_pred:
            try:
                cm_all_count = confusion_matrix(all_true, all_pred, labels=LABEL_ORDER)
                cm_all_norm = confusion_matrix(
                    all_true, all_pred, labels=LABEL_ORDER, normalize="true"
                )

                plot_cm(
                    cm_all_count,
                    "Confusion Matrix (Count)\nALL VIDEOS",
                    all_dir / "all_videos_count.png",
                    normalized=False,
                    experiment_name=exp_name
                )

                plot_cm(
                    cm_all_norm,
                    "Confusion Matrix (Normalized)\nALL VIDEOS",
                    all_dir / "all_videos_normalized.png",
                    normalized=True,
                    experiment_name=exp_name
                )

                print("✓ Confusion matrix ALL VIDEOS dibuat")
            except Exception as e:
                print(f"[ERROR] Gagal membuat confusion matrix ALL VIDEOS: {e}")

    print("\n🎉 Semua confusion matrix comment-level selesai.")


if __name__ == "__main__":
    main()
