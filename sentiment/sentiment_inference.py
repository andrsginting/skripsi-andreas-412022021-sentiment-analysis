# sentiment/sentiment_inference.py
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import softmax

from .model_loader import load_model_and_tokenizer

# Helper: determine pos/neg indices robustly
def _resolve_pos_neg_indices(model):
    """
    Return (pos_idx, neg_idx, neutral_idx_or_None, label_names_list)
    Strategy:
      - If model.config.id2label exists and values are 'LABEL_0'..'LABEL_2' (typical for mdhugol),
        we assume mapping LABEL_0->positive, LABEL_1->neutral, LABEL_2->negative (per repo docs).
      - If more/other labels exist, try to detect keywords in id2label values (not guaranteed).
      - If cannot detect negative/positive indices, raise an informative error.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("Model has no config to read id2label mapping.")

    id2label = getattr(cfg, "id2label", None)
    if not id2label:
        # try str keys
        raise RuntimeError("Model config does not expose id2label mapping.")

    # Build label list in order of indices 0..(n-1)
    n_labels = getattr(cfg, "num_labels", None) or len(id2label)
    labels = []
    for i in range(n_labels):
        key = str(i)
        if key in id2label:
            labels.append(id2label[key])
        else:
            # fallback: try int key
            # some configs may have int keys already
            labels.append(id2label.get(i, f"LABEL_{i}"))

    # Specific rule for mdhugol/indonesia-bert-sentiment-classification:
    # repo docs: LABEL_0->positive, LABEL_1->neutral, LABEL_2->negative
    if n_labels == 3 and all(l.startswith("LABEL_") for l in labels):
        # assume mapping as documented
        pos_idx = labels.index("LABEL_0")
        neu_idx = labels.index("LABEL_1")
        neg_idx = labels.index("LABEL_2")
        return pos_idx, neg_idx, neu_idx, labels

    # Generic attempt: look for keywords in label strings
    low_labels = [l.lower() for l in labels]
    pos_idx = None
    neg_idx = None
    neu_idx = None
    for idx, lname in enumerate(low_labels):
        if any(k in lname for k in ["pos", "positive", "positif", "positiva"]):
            pos_idx = idx
        if any(k in lname for k in ["neg", "negative", "negatif", "negativa"]):
            neg_idx = idx
        if any(k in lname for k in ["neu", "neutral", "netral", "netural"]):
            neu_idx = idx

    # If we found pos & neg indices, return
    if pos_idx is not None and neg_idx is not None:
        return pos_idx, neg_idx, neu_idx, labels

    # Last resort: if 2 labels, assume LABEL_0 negative, LABEL_1 positive (conventional for some)
    if n_labels == 2 and all(l.startswith("LABEL_") for l in labels):
        return 1, 0, None, labels  # pos_idx=1, neg_idx=0

    # Otherwise fail with helpful message
    raise RuntimeError(
        f"Could not automatically resolve positive/negative label indices from model labels: {labels}.\n"
        "If you use a custom/fine-tuned model, ensure its config.id2label is standard or adapt the code."
    )


# Batch scoring function
def compute_sentiment_scores(df, batch_size=64, model_name=None):
    tokenizer, model, device, id2label = load_model_and_tokenizer(model_name=model_name)
    pos_idx, neg_idx, neu_idx, label_names = _resolve_pos_neg_indices(model)

    texts = df["cleaned_comment"].fillna("").astype(str).tolist()
    scores = []
    predicted_labels = []   # <-- tambahan

    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment inference", ncols=80):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs_batch = softmax(logits, dim=-1).cpu().numpy()

        for probs in probs_batch:
            p_pos = float(probs[pos_idx])
            p_neg = float(probs[neg_idx])
            score = max(-1.0, min(1.0, p_pos - p_neg))

            scores.append(score)

            # label kategori
            predicted_labels.append(_label_from_probabilities(probs, label_names))

    df = df.copy()
    df["sentiment_score"] = scores
    df["predicted_label"] = predicted_labels   # <-- tambahan kolom

    return df


def analyze_and_save(csv_path, output_path, batch_size=64, model_name=None):
    print(f"[PROCESS] Analysing file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "cleaned_comment" not in df.columns:
        raise ValueError("Input CSV must contain 'cleaned_comment' column.")
    df_out = compute_sentiment_scores(df, batch_size=batch_size, model_name=model_name)

    # ===== Simpan kolom yang relevan, termasuk konteks =====
    keep_cols = []
    for col in ["thread_id", "cleaned_comment", "likes_count", "is_reply",
            "sentiment_score", "predicted_label"]:
        if col in df_out.columns:
            keep_cols.append(col)

    df_out = df_out[keep_cols]
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved sentiment CSV: {output_path}")



# Interactive single-sentence inference
def infer_single_sentence(text, model_name=None):
    tokenizer, model, device, id2label = load_model_and_tokenizer(model_name=model_name)
    pos_idx, neg_idx, neu_idx, label_names = _resolve_pos_neg_indices(model)
    model.eval()
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    p_pos = float(probs[pos_idx]) if pos_idx < len(probs) else 0.0
    p_neg = float(probs[neg_idx]) if neg_idx < len(probs) else 0.0
    score = max(-1.0, min(1.0, p_pos - p_neg))

    dominant_idx = int(np.argmax(probs))
    # handle both int and str keys
    label_key = model.config.id2label.get(str(dominant_idx), model.config.id2label.get(dominant_idx, "UNKNOWN"))

    label_map_human = {
        "LABEL_0": "positive",
        "LABEL_1": "neutral",
        "LABEL_2": "negative"
    }
    human_label = label_map_human.get(label_key, label_key)

    print("\n[RESULT] Single-sentence inference")
    print(f"  Input: {text}")
    print(f"  Label (raw): {label_key} -> interpreted: {human_label}")
    print(f"  Probabilities: {probs.tolist()}")
    print(f"  Continuous score (pos-neg): {score:.4f}\n")
    return score

def _label_from_probabilities(probs, labels):
    """Mengembalikan label human-readable ('positive','neutral','negative')."""
    dominant_idx = int(np.argmax(probs))
    
    label_key = labels[dominant_idx]  # ex: "LABEL_0"
    label_map_human = {
        "LABEL_0": "positive",
        "LABEL_1": "neutral",
        "LABEL_2": "negative"
    }
    return label_map_human.get(label_key, "unknown")
