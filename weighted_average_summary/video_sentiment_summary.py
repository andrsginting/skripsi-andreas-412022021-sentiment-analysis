# weighted_average_summary/video_sentiment_summary.py
import os
import pandas as pd
from pathlib import Path

BASE_SUMMARY_DIR = Path("sentiment/dataset/summary")
OUTPUT_BASE_DIR = Path("weighted_average_summary")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)


def classify_sentiment(score, pos_th=0.05, neg_th=-0.05):
    if score > pos_th:
        return "positive"
    elif score < neg_th:
        return "negative"
    else:
        return "neutral"


def summarize_video_sentiments():
    experiments = [d for d in BASE_SUMMARY_DIR.iterdir() if d.is_dir()]
    if not experiments:
        print("[INFO] Tidak ada folder eksperimen di sentiment/dataset/summary")
        return

    for exp_dir in experiments:
        print(f"\n📊 MEMPROSES EKSPERIMEN: {exp_dir.name}")

        out_dir = OUTPUT_BASE_DIR / exp_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []

        files = list(exp_dir.glob("*_cleaned_summary.csv"))
        if not files:
            print(f"[WARN] Tidak ada summary CSV di {exp_dir}")
            continue

        for f in files:
            df = pd.read_csv(f)
            if "weighted_avg_sentiment" not in df.columns:
                print(f"[WARN] Kolom weighted_avg_sentiment tidak ada di {f.name}")
                continue

            df["label"] = df["weighted_avg_sentiment"].apply(classify_sentiment)

            total_threads = len(df)
            pos = (df["label"] == "positive").sum()
            neu = (df["label"] == "neutral").sum()
            neg = (df["label"] == "negative").sum()

            results.append({
                "video_name": f.stem.replace("_cleaned_summary", ""),
                "total_threads": total_threads,
                "positive_%": round(pos / total_threads * 100, 2),
                "neutral_%": round(neu / total_threads * 100, 2),
                "negative_%": round(neg / total_threads * 100, 2),
            })

            print(
                f"📹 {f.stem}: "
                f"{pos/total_threads*100:.2f}% positif, "
                f"{neu/total_threads*100:.2f}% netral, "
                f"{neg/total_threads*100:.2f}% negatif"
            )

        df_out = pd.DataFrame(results)
        out_csv = out_dir / "video_sentiment_overview.csv"
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"✅ Disimpan: {out_csv}")


if __name__ == "__main__":
    summarize_video_sentiments()
