# new_llm_judge/thread_evaluation/runners/run_thread_judge.py

import os
import json
import pandas as pd
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# ======================================================
# SETUP PATH
# ======================================================
load_dotenv()

BASE_DIR = os.path.join("new_llm_judge", "thread_evaluation")

THREAD_JSON_DIR = os.path.join(BASE_DIR, "output", "thread_json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thread_labels")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "thread_sentiment_prompt.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 50  # ← sesuai permintaan Anda

# ======================================================
# LOAD PROMPT
# ======================================================
with open(PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# ======================================================
# LLM INITIALIZATION
# ======================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=5
)

# ======================================================
# JUDGE FUNCTION
# ======================================================
def judge_thread(thread_json: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(thread_json, ensure_ascii=False)}
    ]

    response = llm.invoke(messages)
    raw = response.content.lower()

    match = re.search(r"label\s*:\s*(positif|negatif|netral)", raw)
    if match:
        return match.group(1)

    # fallback aman dan eksplisit
    return "netral"

# ======================================================
# FILE SELECTION
# ======================================================
def list_thread_json_files():
    files = sorted(f for f in os.listdir(THREAD_JSON_DIR) if f.endswith(".json"))
    if not files:
        print("[INFO] Tidak ada file thread JSON ditemukan.")
        return []

    print("\n📂 File thread JSON tersedia:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    return files

# ======================================================
# MAIN RUNNER
# ======================================================
def main():
    files = list_thread_json_files()
    if not files:
        return

    choice = input("\nPilih file (contoh: 1,3,5): ").strip()
    selected_idx = [
        int(x) - 1 for x in choice.split(",")
        if x.strip().isdigit() and 0 <= int(x) - 1 < len(files)
    ]

    for idx in selected_idx:
        file_name = files[idx]
        input_path = os.path.join(THREAD_JSON_DIR, file_name)
        output_path = os.path.join(
            OUTPUT_DIR,
            file_name.replace("_threads.json", "_thread_llm.csv")
        )

        print(f"\n▶ Memproses: {file_name}")

        with open(input_path, encoding="utf-8") as f:
            threads = json.load(f)

        total_threads = len(threads)
        print(f"Total thread: {total_threads}")

        # Siapkan file output (write header sekali)
        if not os.path.exists(output_path):
            pd.DataFrame(columns=["thread_id", "llm_thread_label"]).to_csv(
                output_path, index=False, encoding="utf-8-sig"
            )

        # Progress bar
        with tqdm(total=total_threads, desc="Judging threads", unit="thread") as pbar:
            batch_results = []

            for i, thread in enumerate(threads, 1):
                thread_id = thread.get("thread_id", "unknown")
                thread_data = {
                    "main_comment": thread.get("main_comment", ""),
                    "replies": thread.get("replies", [])
                }

                try:
                    label = judge_thread(thread_data)
                except Exception as e:
                    print(f"[LLM ERROR] {thread_id}: {e}")
                    label = "netral"

                batch_results.append({
                    "thread_id": thread_id,
                    "llm_thread_label": label
                })

                # Jika batch penuh atau terakhir
                if len(batch_results) == BATCH_SIZE or i == total_threads:
                    df_batch = pd.DataFrame(batch_results)
                    df_batch.to_csv(
                        output_path,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8-sig"
                    )
                    batch_results.clear()

                pbar.update(1)

        print(f"✓ Output disimpan: {output_path}")

    print("\n🎉 Semua proses thread judging selesai.")

if __name__ == "__main__":
    main()
