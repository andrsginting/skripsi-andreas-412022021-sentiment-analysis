# sentiment/runners/run_thread_aggregation.py
from pathlib import Path
import pandas as pd
from sentiment.aggregation import aggregate_thread_sentiments

BASE_CONTEXTUAL_DIR = Path("sentiment/dataset/contextual")
BASE_SUMMARY_DIR = Path("sentiment/dataset/summary")

def main():
    experiments = [d for d in BASE_CONTEXTUAL_DIR.iterdir() if d.is_dir()]
    if not experiments:
        print("[INFO] Tidak ada folder eksperimen contextual.")
        return

    print("\n📂 Eksperimen tersedia:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp.name}")

    choice = input("\nPilih eksperimen (contoh: 1,3): ").strip()
    exp_idxs = [int(x)-1 for x in choice.split(",") if x.strip().isdigit()]

    for i in exp_idxs:
        if i < 0 or i >= len(experiments):
            continue

        exp_dir = experiments[i]
        summary_dir = BASE_SUMMARY_DIR / exp_dir.name
        summary_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== AGGREGATION {exp_dir.name} ===")

        for csv_file in exp_dir.glob("*.csv"):
            print(f"[PROCESS] {csv_file.name}")

            df = pd.read_csv(csv_file)
            summary = aggregate_thread_sentiments(df)

            out_path = summary_dir / csv_file.name.replace("_contextual.csv", "_cleaned_summary.csv")
            summary.to_csv(out_path, index=False, encoding="utf-8-sig")

            print(f"✓ Saved: {out_path.name}")

    print("\n🎉 Agregasi thread-level selesai.")

if __name__ == "__main__":
    main()
