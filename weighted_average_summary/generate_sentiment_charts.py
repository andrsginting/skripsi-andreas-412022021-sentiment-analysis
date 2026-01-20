# weighted_average_summary/generate_sentiment_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_OUTPUT_DIR = Path("weighted_average_summary")

COLORS = {
    "positive": "#2ECC71",
    "neutral": "#95A5A6",
    "negative": "#E74C3C",
}


def generate_charts_for_experiment(exp_dir: Path):
    csv_path = exp_dir / "video_sentiment_overview.csv"
    if not csv_path.exists():
        print(f"[WARN] File tidak ditemukan: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # ==============================
    # SORT VIDEO
    # ==============================
    df["video_num"] = df["video_name"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("video_num")

    videos = df["video_name"].str.replace("dataset_video_", "Video ")

    # Persentase
    pos_pct = df["positive_%"]
    neu_pct = df["neutral_%"]
    neg_pct = df["negative_%"]

    # Jumlah thread absolut
    total_threads = df["total_threads"]
    pos_cnt = (pos_pct / 100 * total_threads).round().astype(int)
    neu_cnt = (neu_pct / 100 * total_threads).round().astype(int)
    neg_cnt = (neg_pct / 100 * total_threads).round().astype(int)

    # ==============================
    # BAR CHART (PER VIDEO)
    # ==============================
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(videos))
    width = 0.6

    b1 = ax.bar(x, pos_pct, width, label="Positif", color=COLORS["positive"])
    b2 = ax.bar(x, neu_pct, width, bottom=pos_pct, label="Netral", color=COLORS["neutral"])
    b3 = ax.bar(x, neg_pct, width, bottom=pos_pct + neu_pct, label="Negatif", color=COLORS["negative"])

    ax.set_title(
        f"Distribusi Sentimen per Video\nEksperimen {exp_dir.name}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Persentase (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(videos)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ===== TULIS LABEL (% + JUMLAH THREAD) =====
    for i in range(len(videos)):
        if pos_pct.iloc[i] > 4:
            ax.text(
                x[i],
                pos_pct.iloc[i] / 2,
                f"{pos_pct.iloc[i]:.1f}%\n({pos_cnt.iloc[i]})",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold"
            )
        if neu_pct.iloc[i] > 3:
            ax.text(
                x[i],
                pos_pct.iloc[i] + neu_pct.iloc[i] / 2,
                f"{neu_pct.iloc[i]:.1f}%\n({neu_cnt.iloc[i]})",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold"
            )
        if neg_pct.iloc[i] > 4:
            ax.text(
                x[i],
                pos_pct.iloc[i] + neu_pct.iloc[i] + neg_pct.iloc[i] / 2,
                f"{neg_pct.iloc[i]:.1f}%\n({neg_cnt.iloc[i]})",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold"
            )

    bar_path = exp_dir / "bar_chart_sentiment_per_video.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================
    # PIE CHART (KESELURUHAN)
    # ==============================
    total_all_threads = total_threads.sum()

    pos_all = pos_cnt.sum()
    neu_all = neu_cnt.sum()
    neg_all = neg_cnt.sum()

    sizes = [
        pos_all / total_all_threads * 100,
        neu_all / total_all_threads * 100,
        neg_all / total_all_threads * 100,
    ]

    labels = [
        f"Positif\n{pos_all} threads\n({sizes[0]:.1f}%)",
        f"Netral\n{neu_all} threads\n({sizes[1]:.1f}%)",
        f"Negatif\n{neg_all} threads\n({sizes[2]:.1f}%)",
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
        startangle=90,
        explode=(0.05, 0.05, 0.05),
        autopct="%1.1f%%",
        textprops={"fontsize": 11}
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(
        f"Distribusi Sentimen Keseluruhan\nEksperimen {exp_dir.name}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    pie_path = exp_dir / "pie_chart_overall_sentiment.png"
    plt.tight_layout()
    plt.savefig(pie_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Chart lengkap (%, threads) dibuat untuk {exp_dir.name}")


def main():
    experiments = [d for d in BASE_OUTPUT_DIR.iterdir() if d.is_dir()]
    if not experiments:
        print("[INFO] Tidak ada folder eksperimen di weighted_average_summary")
        return

    for exp in experiments:
        print(f"\n📊 Generating charts: {exp.name}")
        generate_charts_for_experiment(exp)

    print("\n🎉 Semua visualisasi berhasil dibuat.")


if __name__ == "__main__":
    main()
