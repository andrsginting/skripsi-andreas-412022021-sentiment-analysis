import os
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ======================================================
# PATH SETUP
# ======================================================
BASE_DIR = Path("evaluation/llm_judge_evaluation")
INDOBERT_BASE = Path("sentiment/dataset/summary")
LLM_DIR = Path("llm_judge/output")

RESULT_DIR = BASE_DIR / "results"

LABEL_ORDER = ["positif", "netral", "negatif"]

# ======================================================
# HELPER
# ======================================================
def majority_vote(labels):
    counter = Counter(labels)
    most_common = counter.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] == most_common[1][1]:
        return "netral"
    return most_common[0][0]


def map_score_to_label(score, pos_th=0.05, neg_th=-0.05):
    if score > pos_th:
        return "positif"
    elif score < neg_th:
        return "negatif"
    return "netral"


# ======================================================
# MAIN
# ======================================================
def main():
    experiments = ["60_main_sentiment", "70_main_sentiment", "80_main_sentiment"]

    for exp in experiments:
        print(f"\n=== EVALUATION LLM COMMENT LEVEL | {exp} ===")

        summary_dir = INDOBERT_BASE / exp
        out_base = RESULT_DIR / exp
        merged_dir = out_base / "merged"
        summary_out = out_base / "summary"

        merged_dir.mkdir(parents=True, exist_ok=True)
        summary_out.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        all_true, all_pred = [], []

        for summary_file in summary_dir.glob("*_summary.csv"):
            video_id = summary_file.stem.replace("_summary", "")
            llm_file = LLM_DIR / f"{video_id.replace('_cleaned','')}_llm.csv"

            if not llm_file.exists():
                print(f"[SKIP] LLM file tidak ditemukan: {llm_file.name}")
                continue

            df_indo = pd.read_csv(summary_file)
            df_llm = pd.read_csv(llm_file)

            # ===============================
            # INDO LABEL
            # ===============================
            df_indo["indo_label"] = df_indo["weighted_avg_sentiment"].apply(
                map_score_to_label
            )

            # ===============================
            # AGREGASI LLM → THREAD LEVEL
            # ===============================
            llm_thread = (
                df_llm
                .groupby("thread_id")["llm_result"]
                .apply(list)
                .apply(majority_vote)
                .reset_index(name="llm_label")
            )

            # ===============================
            # MERGE
            # ===============================
            df_merge = df_indo.merge(llm_thread, on="thread_id", how="inner")

            out_merge = merged_dir / f"{video_id}_merged.csv"
            df_merge.to_csv(out_merge, index=False, encoding="utf-8-sig")

            y_true = df_merge["llm_label"]
            y_pred = df_merge["indo_label"]

            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0
            )

            summary_rows.append({
                "video": video_id,
                "accuracy": acc,
                "macro_precision": p,
                "macro_recall": r,
                "macro_f1": f1,
                "total_threads": len(df_merge)
            })

            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())

        # ===============================
        # SAVE SUMMARY
        # ===============================
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(
            summary_out / "summary_per_video.csv",
            index=False,
            encoding="utf-8-sig"
        )

        acc_all = accuracy_score(all_true, all_pred)
        p_all, r_all, f1_all, _ = precision_recall_fscore_support(
            all_true, all_pred,
            average="macro",
            zero_division=0
        )

        pd.DataFrame([{
            "accuracy": acc_all,
            "macro_precision": p_all,
            "macro_recall": r_all,
            "macro_f1": f1_all
        }]).to_csv(
            summary_out / "overall_metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"✓ Summary evaluasi selesai untuk {exp}")

    print("\n🎉 Semua evaluasi LLM comment-level selesai.")


if __name__ == "__main__":
    main()
