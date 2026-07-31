import pandas as pd
import numpy as np
from pathlib import Path
from config import *


# ==========================================================
# Read CSV
# ==========================================================
def read_csv(file_path):
    df = pd.read_csv(file_path, sep=CSV_SEPARATOR)
    df = df.sort_values(TIME_COLUMN)
    df = df.reset_index(drop=True)
    return df


# ==========================================================
# Convert ms -> second
# ==========================================================
def convert_time(df):
    df = df.copy()
    df["time_s"] = (
        df[TIME_COLUMN] - df[TIME_COLUMN].iloc[0]
    ) / 1000
    return df


# ==========================================================
# Create output folder
# ==========================================================
def create_output_folder(module_name, dataset_name):
    folder = RESULT_DIR / module_name / dataset_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ==========================================================
# Save dataframe
# ==========================================================
def save_csv(df, path):
    df.to_csv(path, index=False)


# ==========================================================
# Save text
# ==========================================================
def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ==========================================================
# Print title
# ==========================================================
def print_title(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Dataset information
# ==========================================================
def dataset_info(df):
    info = {
        "Samples": len(df),
        "Duration (s)": df["time_s"].iloc[-1],
        "Columns": len(df.columns)
    }
    return info
