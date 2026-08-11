import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from config import *

# ==========================
# CONFIG
# ==========================
DATA_DIR = "data/raw"
OUTPUT = "inspection.csv"

# ==========================
# PROCESS
# ==========================
results = []

for file in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    print(f"Processing {os.path.basename(file)}")
    df = pd.read_csv(file, sep=None, engine="python")
    amp = df["amp_mean"].values
    phase = df["phase_std"].values
    amp_std = np.std(amp)
    phase_std = np.std(phase)
    energy = np.sum(amp**2)
    peaks, _ = find_peaks(
        amp,
        prominence=np.std(amp) * 0.5
    )
    peak_count = len(peaks)
    current_label = int(df["label"].mode()[0])
    results.append({
        "File": os.path.basename(file),
        "CurrentLabel": current_label,
        "AmpSTD": amp_std,
        "PhaseSTD": phase_std,
        "Energy": energy,
        "PeakCount": peak_count
    })

# ==========================
# SCORE
# ==========================
result = pd.DataFrame(results)

metrics = [
    "AmpSTD",
    "PhaseSTD",
    "Energy",
    "PeakCount"
]

for col in metrics:
    result[col] = (
        result[col] - result[col].min()
    ) / (
        result[col].max() - result[col].min() + 1e-8
    )

result["MotionScore"] = (
    result["AmpSTD"] +
    result["PhaseSTD"] +
    result["Energy"] +
    result["PeakCount"]
) / 4

result["SuggestedLabel"] = np.where(
    result["MotionScore"] > 0.5,
    1,
    0
)

label_counts = result["SuggestedLabel"].value_counts()
static_count = int(label_counts.get(0, 0))
motion_count = int(label_counts.get(1, 0))

result = result.sort_values(
    "MotionScore",
    ascending=False
)

result.to_csv(
    OUTPUT,
    sep=CSV_SEPARATOR,
    index=False
)

print()
print(result)

print("\nSuggested label counts:")
print(f"  Static (0): {static_count}")
print(f"  Motion (1): {motion_count}")

print(f"\nSaved to {OUTPUT}")
