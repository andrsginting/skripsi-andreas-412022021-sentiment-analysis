# check_batch.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATCH_FOLDER = "llm_judge/batch_jobs/"

def main():
    print("\n=== CHECKING ALL BATCH JOBS ===\n")

    # cari semua file batch_id
    batch_files = [f for f in os.listdir(BATCH_FOLDER) if f.endswith("_batch_id.txt")]
    if not batch_files:
        print("Tidak ada file batch_id ditemukan di llm_judge/batch_jobs/")
        return

    for bid_file in batch_files:
        batch_id = open(os.path.join(BATCH_FOLDER, bid_file)).read().strip()
        print(f"→ Batch ID: {batch_id}")

        try:
            batch = client.batches.retrieve(batch_id)
            print(f"  Status   : {batch.status}")
            print(f"  Created  : {batch.created_at}")
            print(f"  Requests : {batch.request_counts}\n")
        except Exception as e:
            print(f"  ERROR membaca batch: {e}\n")

if __name__ == "__main__":
    main()
