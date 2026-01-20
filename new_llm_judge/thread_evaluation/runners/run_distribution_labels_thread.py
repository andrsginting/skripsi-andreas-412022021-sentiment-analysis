# new_llm_judge/thread_evaluation/runners/run_distribution_labels_thread.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ======================================================
# PATH SETUP
# ======================================================
BASE_DIR = Path("new_llm_judge/thread_evaluation")
LABEL_DIR = BASE_DIR / "output" / "thread_labels"
OUT_DIR = BASE_DIR / "output" / "thread_distribution"

OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["positif", "netral", "negatif"]
COLORS = {
    "positif": "#2ECC71",
    "netral": "#95A5A6",
    "negatif": "#E74C3C"
}

# ======================================================
# MAIN LOGIC
# ======================================================
def main():
    files = sorted(LABEL_DIR.glob("*_thread_llm.csv"))
    if not files:
        print("[INFO] Tidak ada file thread_llm ditemukan.")
        return

    rows = []

    # ==================================================
    # PER VIDEO DISTRIBUTION
    # ==================================================
    for f in files:
        df = pd.read_csv(f)

        if "llm_thread_label" not in df.columns:
            print(f"[SKIP] Kolom llm_thread_label tidak ada: {f.name}")
            continue

        total = len(df)
        counts = df["llm_thread_label"].value_counts()

        pos = counts.get("positif", 0)
        neu = counts.get("netral", 0)
        neg = counts.get("negatif", 0)

        rows.append({
            "video_name": f.stem.replace("_thread_llm", ""),
            "total_threads": total,
            "positive_%": round(pos / total * 100, 2),
            "neutral_%": round(neu / total * 100, 2),
            "negative_%": round(neg / total * 100, 2),
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg
        })

    df_out = pd.DataFrame(rows)

    # Sort video 1–6
    df_out["video_num"] = df_out["video_name"].str.extract("(\d+)").astype(int)
    df_out = df_out.sort_values("video_num")
    df_out.drop(columns="video_num", inplace=True)

    # ==================================================
    # SAVE CSV
    # ==================================================
    csv_path = OUT_DIR / "llm_thread_distribution.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ CSV distribusi disimpan: {csv_path}")

    # ==================================================
    # BAR CHART (PER VIDEO)
    # ==================================================
    videos = df_out["video_name"].str.replace("dataset_video_", "Video ")
    pos = df_out["positive_%"]
    neu = df_out["neutral_%"]
    neg = df_out["negative_%"]

    x = np.arange(len(videos))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(x, pos, width, label="Positif", color=COLORS["positif"])
    ax.bar(x, neu, width, bottom=pos, label="Netral", color=COLORS["netral"])
    ax.bar(x, neg, width, bottom=pos + neu, label="Negatif", color=COLORS["negatif"])

    # Annotation (persentase + jumlah thread)
    for i in range(len(videos)):
        ax.text(i, pos[i] / 2, f"{pos[i]:.1f}%\n({df_out.iloc[i]['positive_count']})",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        ax.text(i, pos[i] + neu[i] / 2,
                f"{neu[i]:.1f}%\n({df_out.iloc[i]['neutral_count']})",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        ax.text(i, pos[i] + neu[i] + neg[i] / 2,
                f"{neg[i]:.1f}%\n({df_out.iloc[i]['negative_count']})",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_title("Distribusi Sentimen per Video\n(Ground Truth GPT - Thread Level)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Persentase (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(videos)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    bar_path = OUT_DIR / "bar_chart_llm_per_video.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"✓ Bar chart disimpan: {bar_path}")

    # ==================================================
    # PIE CHART (OVERALL)
    # ==================================================
    total_threads = df_out["total_threads"].sum()

    total_pos = df_out["positive_count"].sum()
    total_neu = df_out["neutral_count"].sum()
    total_neg = df_out["negative_count"].sum()

    sizes = [
        total_pos / total_threads * 100,
        total_neu / total_threads * 100,
        total_neg / total_threads * 100
    ]

    labels = [
        f"Positif\n{total_pos} threads\n({sizes[0]:.1f}%)",
        f"Netral\n{total_neu} threads\n({sizes[1]:.1f}%)",
        f"Negatif\n{total_neg} threads\n({sizes[2]:.1f}%)"
    ]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.pie(
        sizes,
        labels=labels,
        colors=[COLORS["positif"], COLORS["netral"], COLORS["negatif"]],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0.05, 0.05),
        textprops={"fontsize": 11, "fontweight": "bold"}
    )

    ax.set_title(
        "Distribusi Sentimen Semua Video\n(Ground Truth GPT – Thread Level)",
        fontsize=14,
        fontweight="bold"
    )

    pie_path = OUT_DIR / "pie_chart_llm_overall.png"
    plt.tight_layout()
    plt.savefig(pie_path, dpi=300)
    plt.close()
    print(f"✓ Pie chart disimpan: {pie_path}")

    print("\n🎉 Distribusi label LLM selesai.")

# ======================================================
if __name__ == "__main__":
    main()
