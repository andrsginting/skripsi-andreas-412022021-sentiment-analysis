import re
import pandas as pd  # ← INI WAJIB ADA

def count_words(text):
    """
    Menghitung jumlah kata dalam komentar.
    Regex memastikan hanya karakter alfanumerik dihitung sebagai kata.
    """
    if pd.isna(text):
        return 0

    # split kata berdasarkan regex
    tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    return len(tokens)
