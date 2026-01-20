# sentiment/aggregation.py
import pandas as pd


def compute_weighted_thread_sentiment(
    df,
    like_weight=0.7,
    reply_weight=0.3
):
    required = {"thread_id", "contextual_score", "likes_count", "is_reply"}
    if not required.issubset(df.columns):
        raise ValueError(f"Kolom wajib: {required}")

    df = df.copy()
    df["likes_count"] = df["likes_count"].fillna(0).astype(float)

    df["weight"] = (
        like_weight * (df["likes_count"] + 1)
        + reply_weight * df["is_reply"].astype(int)
    )

    df["weighted_score"] = df["contextual_score"] * df["weight"]

    summary = (
        df.groupby("thread_id")
        .agg(
            total_weight=("weight", "sum"),
            weighted_sum=("weighted_score", "sum"),
            total_comments=("thread_id", "count")
        )
        .reset_index()
    )

    summary["weighted_avg_sentiment"] = (
        summary["weighted_sum"] / summary["total_weight"]
    ).clip(-1, 1)

    return summary


def compute_overall_sentiment(summary):
    total_weight = summary["total_weight"].sum()
    if total_weight == 0:
        return 0.0

    return (
        (summary["weighted_avg_sentiment"] * summary["total_weight"]).sum()
        / total_weight
    )


def aggregate_thread_sentiments(
    csv_path_or_df,
    like_weight=0.7,
    reply_position_weight=0.3
):
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    summary = compute_weighted_thread_sentiment(
        df,
        like_weight=like_weight,
        reply_weight=reply_position_weight
    )

    overall = compute_overall_sentiment(summary)
    print(f"[RESULT] Overall sentiment: {overall:.4f}")

    return summary
