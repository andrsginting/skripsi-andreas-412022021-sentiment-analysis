# new_evaluation/thread_evaluation/runners/run_thread_evaluation.py
import os
import pandas as pd
import sys
from pathlib import Path

# ======================================================
# FIX PYTHON PATH (PROJECT ROOT)
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from new_evaluation.thread_evaluation.utils.label_mapper import map_score_to_label
from new_evaluation.thread_evaluation.utils.metrics import compute_metrics, build_confusion_matrix

LABEL_ORDER = ["positif", "netral", "negatif"]

INDOBERT_BASE = Path("sentiment/dataset/summary")
LLM_DIR = Path("new_llm_judge/thread_evaluation/output/thread_labels")
RESULT_BASE = Path("new_evaluation/thread_evaluation/results")


def main():
    experiments = [d for d in INDOBERT_BASE.iterdir() if d.is_dir()]
    if not experiments:
        print("[INFO] Tidak ada folder eksperimen IndoBERT.")
        return

    print("\n📂 Eksperimen tersedia:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp.name}")

    choice = input("\nPilih eksperimen (contoh: 1,3 atau kosong untuk semua): ").strip()
    if choice:
        idxs = [int(x)-1 for x in choice.split(",") if x.strip().isdigit()]
        selected_exps = [experiments[i] for i in idxs if 0 <= i < len(experiments)]
    else:
        selected_exps = experiments

    for exp_dir in selected_exps:
        print(f"\n=== EVALUATION EXPERIMENT: {exp_dir.name} ===")

        # === OUTPUT DIR ===
        base_out = RESULT_BASE / exp_dir.name
        merged_dir = base_out / "merged"
        per_video_dir = base_out / "per_video"
        summary_dir = base_out / "summary"

        for d in [merged_dir, per_video_dir, summary_dir]:
            d.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        all_true, all_pred = [], []

        for indo_file in exp_dir.glob("*_cleaned_summary.csv"):
            raw_video_id = indo_file.stem.replace("_cleaned_summary", "")
            video_id = raw_video_id.replace("_cleaned", "")

            llm_file = LLM_DIR / f"{video_id}_thread_llm.csv"

            if not llm_file.exists():
                print(f"[SKIP] Ground truth tidak ditemukan: {llm_file.name}")
                continue

            print(f"\n▶ Evaluasi video: {video_id}")

            df_indo = pd.read_csv(indo_file)
            df_llm = pd.read_csv(llm_file)

            df_indo["indo_label"] = df_indo["weighted_avg_sentiment"].apply(map_score_to_label)

            df_merge = df_indo.merge(df_llm, on="thread_id", how="inner")

            df_merge = df_merge[
                ["thread_id", "weighted_avg_sentiment", "indo_label", "llm_thread_label", "total_comments"]
            ]

            # === SAVE MERGED ===
            merge_path = merged_dir / f"{video_id}_merged.csv"
            df_merge.to_csv(merge_path, index=False, encoding="utf-8-sig")

            print(f"✓ Merged → {merge_path.name}")
            print(f"  Total thread dievaluasi: {len(df_merge)}")

            # === METRICS ===
            y_true = df_merge["llm_thread_label"]
            y_pred = df_merge["indo_label"]

            metrics = compute_metrics(y_true, y_pred)
            cm = build_confusion_matrix(y_true, y_pred)

            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{l}" for l in LABEL_ORDER],
                columns=[f"pred_{l}" for l in LABEL_ORDER]
            )

            cm_path = per_video_dir / f"{video_id}_confusion_matrix.csv"
            cm_df.to_csv(cm_path, encoding="utf-8-sig")

            summary_rows.append({
                "video": video_id,
                **metrics,
                "total_threads": len(df_merge)
            })

            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())

        # === SUMMARY PER EXPERIMENT ===
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_dir / "summary_per_video.csv", index=False)

        overall_metrics = compute_metrics(all_true, all_pred)
        overall_df = pd.DataFrame([overall_metrics])
        overall_df.to_csv(summary_dir / "overall_metrics.csv", index=False)

        print(f"\n✓ Summary eksperimen {exp_dir.name} selesai.")

    print("\n🎉 Semua evaluasi thread-level selesai.")


if __name__ == "__main__":
    main()
