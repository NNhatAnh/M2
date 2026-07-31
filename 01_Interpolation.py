import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
from utils import read_csv


def convert_time(df):
    df = df.copy()
    df["time_s"] = (df[TIME_COLUMN] - df[TIME_COLUMN].iloc[0]) / 1000
    return df


def analyze_packet_interval(df):
    dt = np.diff(df["time_s"])
    stats = {
        "duration": df["time_s"].iloc[-1],
        "samples": len(df),
        "mean_dt": dt.mean(),
        "std_dt": dt.std(),
        "mean_fs": 1/dt.mean(),
        "std_fs": np.std(1/dt)
    }
    return dt, stats


def linear_resample(df):
    t_uniform = np.arange(df["time_s"].iloc[0],
                          df["time_s"].iloc[-1], 1 / FS_TARGET)
    out = pd.DataFrame()
    out["time_s"] = t_uniform

    for feature in FEATURE_COLUMNS:
        out[feature] = np.interp(
            t_uniform,
            df["time_s"],
            df[feature]
        )
    out["label"] = df["label"].iloc[0]
    return out


def plot_packet_interval(dt):
    plt.figure(figsize=(7, 4))
    plt.hist(dt*1000, bins=30, edgecolor='black')
    plt.xlabel("Packet Interval (ms)")
    plt.ylabel("Count")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(INTERP_DIR / "01_packet_interval_histogram.png", dpi=300)
    plt.close()


def plot_sampling_frequency(dt):
    fs = 1/dt
    plt.figure(figsize=(8, 4))
    plt.plot(fs)
    plt.axhline(FS_TARGET, color='red', linestyle='--')
    plt.ylabel("Sampling Frequency (Hz)")
    plt.xlabel("Packet")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(INTERP_DIR / "02_sampling_frequency.png", dpi=300)
    plt.close()


def plot_interpolation(raw, resampled):
    plt.figure(figsize=(10, 4))
    plt.scatter(raw["time_s"], raw["amp_mean"], s=18, label="Raw")
    plt.plot(resampled["time_s"], resampled["amp_mean"],
             color="red", linewidth=2, label="Linear")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(INTERP_DIR / "03_raw_vs_interpolation.png", dpi=300)
    plt.close()


def plot_zoom(raw, resampled):
    plt.figure(figsize=(10, 4))
    plt.xlim(2, 5)
    plt.scatter(
        raw["time_s"],
        raw["amp_mean"]
    )
    plt.plot(
        resampled["time_s"],
        resampled["amp_mean"],
        color="red"
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(INTERP_DIR / "04_zoom_comparison.png", dpi=300)
    plt.close()


def save_statistics(stats):
    with open(INTERP_DIR / "statistics.txt", "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    for dataset_name, dataset_path in DATASETS.items():
        df = read_csv(dataset_path)
    df = convert_time(df)
    dt, stats = analyze_packet_interval(df)
    resampled = linear_resample(df)
    plot_packet_interval(dt)
    plot_sampling_frequency(dt)
    plot_interpolation(df, resampled)
    plot_zoom(df, resampled)
    save_statistics(stats)
    resampled.to_csv(INTERP_DIR / "interpolated_signal.csv", index=False)
