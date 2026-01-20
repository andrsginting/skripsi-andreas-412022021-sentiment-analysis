import os
import pandas as pd

def load_clean_files(dataset_path="cleaning/dataset"):
    """
    Memuat seluruh file CSV hasil cleaning.
    Mengembalikan list tuple: (filename, dataframe)
    """
    files = []
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{dataset_path}/{file}")
            files.append((file.replace(".csv", ""), df))
    return files
