# new_llm_judge/thread_evaluation/builders/build_thread_json.py

import os
import json
import pandas as pd
from collections import defaultdict

INPUT_DIR = "cleaning/dataset"
OUTPUT_DIR = "new_llm_judge/thread_evaluation/output/thread_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_thread_json(csv_file):
    df = pd.read_csv(csv_file)

    threads = defaultdict(lambda: {
        "thread_id": None,
        "main_comment": None,
        "replies": []
    })

    for _, row in df.iterrows():
        tid = row["thread_id"]

        threads[tid]["thread_id"] = tid

        if not row["is_reply"]:
            threads[tid]["main_comment"] = {
                "comment": row["cleaned_comment"],
                "likes": int(row["likes_count"])
            }
        else:
            threads[tid]["replies"].append({
                "comment": row["cleaned_comment"],
                "likes": int(row["likes_count"])
            })

    return list(threads.values())


def process_dataset(csv_path):
    video_name = os.path.basename(csv_path).replace("_cleaned.csv", "")
    threads = build_thread_json(csv_path)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_threads.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(threads, f, ensure_ascii=False, indent=2)

    print(f"✓ Thread JSON dibuat: {out_path}")


if __name__ == "__main__":
    for f in os.listdir(INPUT_DIR):
        if f.endswith("_cleaned.csv"):
            process_dataset(os.path.join(INPUT_DIR, f))
