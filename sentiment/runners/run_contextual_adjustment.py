# sentiment/runners/run_contextual_adjustment.py
from pathlib import Path
import pandas as pd
from sentiment.contextual_inference import adjust_sentiment_contextually

INPUT_DIR = Path("sentiment/dataset/sentiment")

EXPERIMENTS = {
    "60_main_sentiment": (0.6, 0.4),
    "70_main_sentiment": (0.7, 0.3),
    "80_main_sentiment": (0.8, 0.2),
}

def main():
    files = list(INPUT_DIR.glob("*.csv"))
    if not files:
        print("[INFO] Tidak ada file sentiment untuk diproses.")
        return

    print("\n📂 File sentiment tersedia:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f.name}")

    choice = input("\nPilih file (contoh: 1,2): ").strip()
    idxs = [int(x)-1 for x in choice.split(",") if x.strip().isdigit()]

    for exp_name, (reply_w, main_w) in EXPERIMENTS.items():
        print(f"\n=== EXPERIMENT {exp_name} ===")
        out_dir = Path(f"sentiment/dataset/contextual/{exp_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in idxs:
            if i < 0 or i >= len(files):
                continue

            f = files[i]
            df = pd.read_csv(f)

            print(f"[PROCESS] {f.name}")
            df_ctx = adjust_sentiment_contextually(
                df,
                reply_weight=reply_w,
                main_weight=main_w
            )

            out_path = out_dir / f.name.replace("_sentiment.csv", "_contextual.csv")
            df_ctx.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✓ Saved: {out_path.name}")

    print("\n🎉 Contextual adjustment selesai.")

if __name__ == "__main__":
    main()
