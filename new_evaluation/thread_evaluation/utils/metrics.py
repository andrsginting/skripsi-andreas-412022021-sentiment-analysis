from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import pandas as pd

LABEL_ORDER = ["positif", "netral", "negatif"]

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1
    }

def build_confusion_matrix(y_true, y_pred):
    return confusion_matrix(
        y_true, y_pred,
        labels=LABEL_ORDER
    )
