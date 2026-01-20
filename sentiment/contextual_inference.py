# sentiment/contextual_inference.py
import pandas as pd
import numpy as np


def adjust_sentiment_contextually(
    df_or_path,
    reply_weight=0.6,
    main_weight=0.4
):
    if not np.isclose(reply_weight + main_weight, 1.0):
        raise ValueError("reply_weight + main_weight harus = 1.0")

    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    required = {"thread_id", "sentiment_score", "is_reply"}
    if not required.issubset(df.columns):
        raise ValueError(f"Kolom wajib: {required}")

    print(f"[PROCESS] Contextual adjustment reply={reply_weight}, main={main_weight}")

    # =====================================================
    # 1️⃣ Ambil SENTIMEN KOMENTAR UTAMA (SATU PER THREAD)
    # =====================================================
    main_sentiment = (
        df[df["is_reply"] == False]
        .drop_duplicates(subset=["thread_id"])
        .set_index("thread_id")["sentiment_score"]
        .rename("main_sentiment")
    )

    df = df.merge(main_sentiment, on="thread_id", how="left")

    # =====================================================
    # 2️⃣ Contextual adjustment (HANYA UNTUK BALASAN)
    # =====================================================
    df["contextual_score"] = df["sentiment_score"]

    mask_reply = df["is_reply"] == True
    df.loc[mask_reply, "contextual_score"] = (
        reply_weight * df.loc[mask_reply, "sentiment_score"]
        + main_weight * df.loc[mask_reply, "main_sentiment"]
    )

    # =====================================================
    # 3️⃣ LOG DEBUG (SANGAT DISARANKAN)
    # =====================================================
    print("  ↳ Distribusi is_reply:")
    print(df["is_reply"].value_counts())

    return df
