# llm_judge/runners/run_llm_judge.py

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# FIX import path
sys.path.append(os.path.abspath("."))

from llm_judge.chains.sentiment_chain import build_sentiment_judge

load_dotenv()

INPUT_DIR = "cleaning/dataset/"
OUTPUT_DIR = "llm_judge/output/"
PROMPT_PATH = "llm_judge/prompts/sentiment_prompt.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_datasets():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith("_cleaned.csv")])
    print("\n📂 Dataset tersedia:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    return files


def run_dataset_mode(judge):
    files = list_datasets()
    choice = input("\nPilih dataset (contoh: 1,3,5): ").strip()

    selected_idx = [
        int(x) - 1 for x in choice.split(",")
        if x.strip().isdigit() and 0 < int(x) <= len(files)
    ]

    for idx in selected_idx:
        file = files[idx]
        print(f"\n▶ Memproses: {file}")

        df = pd.read_csv(os.path.join(INPUT_DIR, file))
        out_path = os.path.join(
            OUTPUT_DIR, file.replace("_cleaned.csv", "_llm.csv")
        )

        results = []
        buffer = []

        for i, row in df.iterrows():
            comment = str(row["cleaned_comment"])

            try:
                label = judge(comment)
            except Exception as e:
                print("LLM error:", e)
                label = "netral"

            row_out = {
                "thread_id": row["thread_id"],
                "cleaned_comment": comment,
                "likes_count": row["likes_count"],
                "is_reply": row["is_reply"],
                "llm_result": label
            }

            buffer.append(row_out)

            # ✨ TULIS SETIAP 50 DATA
            if (i + 1) % 50 == 0:
                pd.DataFrame(buffer).to_csv(
                    out_path,
                    mode="a",
                    header=not os.path.exists(out_path),
                    index=False
                )
                print(f"  ✓ {i + 1} komentar tersimpan...")
                buffer = []

        # sisa buffer
        if buffer:
            pd.DataFrame(buffer).to_csv(
                out_path,
                mode="a",
                header=not os.path.exists(out_path),
                index=False
            )

        print("✓ Selesai →", out_path)


def run_interactive_mode(judge):
    print("\n🧪 Mode uji coba manual (ketik 'exit' untuk keluar)")
    while True:
        text = input("\nKomentar: ").strip()
        if text.lower() in ["exit", "quit"]:
            break
        print("→ Sentimen:", judge(text))


def main():
    prompt_text = open(PROMPT_PATH, encoding="utf-8").read()
    judge = build_sentiment_judge(prompt_text)

    print("\n=== MODE ANOTATOR SENTIMEN (GPT-4o-mini) ===")
    print("1️⃣ Analisis dataset CSV")
    print("2️⃣ Uji coba langsung di terminal")

    mode = input("\nPilih mode [1/2]: ").strip()

    if mode == "1":
        run_dataset_mode(judge)
    elif mode == "2":
        run_interactive_mode(judge)
    else:
        print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
