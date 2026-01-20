# evaluation/run_evaluate_indobert_vs_llm.py
import os
import glob
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

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


def normalize(label):
    return LABEL_MAP.get(str(label).strip().lower(), None)


def extract_video_id(filename):
    """
    dataset_video_1_cleaned_sentiment.csv
    -> dataset_video_1
    """
    return filename.split("_cleaned")[0]


def evaluate_pair(video_id, indo_path, llm_path):
    df_indo = pd.read_csv(indo_path)
    df_llm = pd.read_csv(llm_path)

    df = df_indo.merge(
        df_llm,
        on=["thread_id", "cleaned_comment", "likes_count", "is_reply"],
        how="inner"
    )

    if df.empty:
        print(f"[WARN] Tidak ada data cocok untuk {video_id}")
        return None, None

    y_true = df["predicted_label"].apply(normalize)
    y_pred = df["llm_result"].apply(normalize)

    df = df[y_true.notnull() & y_pred.notnull()]
    y_true = df["predicted_label"].apply(normalize)
    y_pred = df["llm_result"].apply(normalize)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, digits=4, zero_division=0
    )

    return {
        "video": video_id,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }, report


def main():
    summaries = []
    all_true, all_pred = [], []

    indo_files = glob.glob(os.path.join(INDOBERT_DIR, "*_sentiment.csv"))

    for indo_path in indo_files:
        filename = os.path.basename(indo_path)
        video_id = extract_video_id(filename)

        llm_candidates = glob.glob(
            os.path.join(LLM_DIR, f"{video_id}*_llm.csv")
        )

        if not llm_candidates:
            print(f"[SKIP] LLM file tidak ditemukan untuk {video_id}")
            continue

        llm_path = llm_candidates[0]

        print(f"✓ Evaluating {video_id}")

        summary, report = evaluate_pair(video_id, indo_path, llm_path)

        if summary:
            summaries.append(summary)

            with open(
                os.path.join(OUT_DIR, f"{video_id}_classification_report.txt"),
                "w"
            ) as f:
                f.write(report)

            df_indo = pd.read_csv(indo_path)
            df_llm = pd.read_csv(llm_path)

            df_merge = df_indo.merge(
                df_llm,
                on=["thread_id", "cleaned_comment", "likes_count", "is_reply"],
                how="inner"
            )

            all_true.extend(df_merge["predicted_label"].apply(normalize))
            all_pred.extend(df_merge["llm_result"].apply(normalize))

    # ===============================
    # SAVE SUMMARY PER VIDEO
    # ===============================
    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(OUT_DIR, "summary_per_video.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== SUMMARY PER VIDEO ===")
    print(summary_df)

    # ===============================
    # OVERALL EVALUATION
    # ===============================
    if all_true and all_pred:
        report_all = classification_report(
            all_true, all_pred, digits=4, zero_division=0
        )

        with open(
            os.path.join(OUT_DIR, "overall_classification_report.txt"),
            "w"
        ) as f:
            f.write(report_all)

        print("\n=== OVERALL EVALUATION (ALL VIDEOS) ===")
        print(report_all)

    print("\n✓ Saved results to evaluation/results/")


if __name__ == "__main__":
    main()
