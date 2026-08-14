"""
==========================================================
MODULE 05 - MOTION EXTRACTION
==========================================================

Input
------
results/background/
    butterworth/
    median/
    moving_average/

Output
-------
results/motion_extraction/
    butterworth/
    median/
    moving_average/

==========================================================
"""

import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from config import *
from utils import *
from metrics import *
from visualization import *
from report import *


# ==========================================================
# Rolling STD
# ==========================================================
def rolling_std(signal):
    signal = pd.Series(signal)
    score = signal.rolling(
        window=ROLLING_WINDOW,
        center=True,
        min_periods=1
    ).std()
    score = score.fillna(0)
    return score.values


# ==========================================================
# Gaussian Smooth
# ==========================================================
def smooth(score):
    return gaussian_filter1d(
        score,
        sigma=GAUSSIAN_SIGMA
    )


# ==========================================================
# Normalize
# ==========================================================
def normalize(score):
    score = np.asarray(score)
    minimum = np.min(score)
    maximum = np.max(score)
    return (
        score-minimum
    ) / (
        maximum-minimum+1e-12
    )


# ==========================================================
# Binary Detection
# ==========================================================
def binary_detection(score):
    threshold = (np.mean(score) + THRESHOLD_FACTOR*np.std(score))
    binary = (score > threshold).astype(np.uint8)
    return binary, threshold


# ==========================================================
# Motion Pipeline
# ==========================================================
class MotionPipeline:
    def __init__(self):
        self.module_name = "motion_extraction"
        self.summary = []

    # ======================================================
    # Load
    # ======================================================
    def load_dataset(self, method, dataset_name):
        input_file = (BACKGROUND_DIR / method / dataset_name / "motion.csv")

        if not input_file.exists():
            raise FileNotFoundError(
                input_file
            )
        df = pd.read_csv(input_file)
        return df

    # ======================================================
    # Process
    # ======================================================
    def process(self, method, dataset_name):
        print_title(
            f"{method.upper()} : {dataset_name.upper()}"
        )
        df = self.load_dataset(
            method,
            dataset_name
        )
        output_folder = (
            MOTION_DIR
            /
            method
            /
            dataset_name
        )
        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )
        score_df = pd.DataFrame()
        binary_df = pd.DataFrame()
        score_df["time_s"] = df["time_s"]
        binary_df["time_s"] = df["time_s"]
        if LABEL_COLUMN in df.columns:
            score_df[LABEL_COLUMN] = df[LABEL_COLUMN]
            binary_df[LABEL_COLUMN] = df[LABEL_COLUMN]
        report = Report()
        report.title(
            f"{method.upper()} - {dataset_name.upper()}"
        )
        metrics_result = []
        start = time.perf_counter()

        # ==================================================
        # Process each feature
        # ==================================================
        for feature in FEATURE_COLUMNS:
            print(f"Processing : {feature}")
            motion = df[feature].values

            score = rolling_std(
                motion
            )

            score = smooth(
                score
            )

            score = np.maximum(
                score - np.percentile(score, 10),
                0
            )
            score = normalize(
                score
            )

            binary, threshold = binary_detection(
                score
            )

            score_df[feature] = score
            binary_df[feature] = binary

            activity_ratio = np.mean(binary) * 100
            motion_duration = np.sum(binary) / TARGET_FS

            peaks, properties = find_peaks(
                score,
                height=threshold,
                distance=ROLLING_WINDOW
            )
            peak_count = len(peaks)

            if len(peaks) > 0:
                mean_peak_height = np.mean(
                    properties["peak_heights"]
                )
            else:
                mean_peak_height = 0

            score_std = np.std(
                score
            )
            score_mean = np.mean(
                score
            )
            metrics_result.append({
                "Feature": feature,
                "Threshold": threshold,
                "Activity Ratio": activity_ratio,
                "Motion Duration": motion_duration,
                "Peak Count": peak_count,
                "Mean Peak Height": mean_peak_height,
                "Score Mean": score_mean,
                "Score STD": score_std
            })

            # ------------------------------------------
            # Report
            # ------------------------------------------
            report.add(
                feature,
                ""
            )
            report.add(
                "Threshold",
                round(
                    threshold,
                    4
                )
            )
            report.add(
                "Activity Ratio",
                round(
                    activity_ratio,
                    4
                )
            )
            report.add(
                "Motion Duration",
                round(
                    motion_duration,
                    4
                )
            )
            report.add(
                "Peak Count",
                int(
                    peak_count
                )
            )
            report.add(
                "Peak Height",
                round(
                    mean_peak_height,
                    4
                )
            )
            report.add(
                "Score Mean",
                round(
                    score_mean,
                    4
                )
            )
            report.add(
                "Score STD",
                round(
                    score_std,
                    4
                )
            )
            report.blank()

        # ==================================================
        # Save CSV
        # ==================================================
        score_file = (
            output_folder
            /
            "motion_score.csv"
        )
        binary_file = (
            output_folder
            /
            "motion_binary.csv"
        )
        save_csv(
            score_df,
            score_file
        )
        save_csv(
            binary_df,
            binary_file
        )

        # ==================================================
        # Save Summary
        # ==================================================
        summary = pd.DataFrame(
            metrics_result
        )
        summary_file = (
            output_folder
            /
            "summary.csv"
        )
        save_csv(
            summary,
            summary_file
        )

        # ==================================================
        # Save Report
        # ==================================================
        report_file = (
            output_folder
            /
            "report.txt"
        )
        report.save(
            report_file
        )

        # ==================================================
        # Plot
        # ==================================================
        figure_folder = (
            output_folder
            / "figures"
        )

        figure_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        x = df["time_s"]
        for feature in FEATURE_COLUMNS:
            plt.figure(figsize=(12, 8))

            # ------------------------------------------
            # Original Motion
            # ------------------------------------------
            plt.subplot(311)
            plt.plot(x, df[feature], linewidth=1)
            plt.title(
                f"{feature} - Residual Motion"
            )
            plt.grid(alpha=0.3)

            # ------------------------------------------
            # Motion Score
            # ------------------------------------------
            plt.subplot(312)
            plt.plot(x, score_df[feature], color="red", linewidth=1.5)
            plt.title("Average Motion Score")
            plt.grid(alpha=0.3)
            plt.scatter(x[peaks], score[peaks], c="red", s=20, label="Peaks")
            plt.legend()

            # ------------------------------------------
            # Binary Motion
            # ------------------------------------------
            plt.subplot(313)
            plt.step(x, binary_df[feature], where="post", color="green")
            plt.ylim(-0.2, 1.2)
            plt.title("Motion Detection")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(figure_folder / f"{feature}.png", dpi=300)
            plt.close()

        # ==================================================
        # Benchmark
        # ==================================================
        self.summary.append({
            "Method": method,
            "Dataset": dataset_name,
            "Mean Activity Ratio": summary["Activity Ratio"].mean(),
            "Mean Motion Duration": summary["Motion Duration"].mean(),
            "Mean Peak Count": summary["Peak Count"].mean(),
            "Mean Peak Height": summary["Mean Peak Height"].mean(),
            "Mean Score STD": summary["Score STD"].mean()
        })

        print()
        print(f"Finished : {method} - {dataset_name}")
        print(f"Output : {output_folder}")
        print()

    # ======================================================
    # Run
    # ======================================================
    def run(self):
        print_title("MODULE 05 : MOTION EXTRACTION")

        for method in METHODS:
            for dataset_name in DATASETS.keys():
                self.process(method, dataset_name)

        # ==================================================
        # Benchmark
        # ==================================================
        benchmark = pd.DataFrame(
            self.summary
        )
        benchmark_folder = (
            BENCHMARK_DIR
        )
        benchmark_folder.mkdir(
            parents=True,
            exist_ok=True
        )
        benchmark_file = (benchmark_folder / "motion_benchmark.csv")
        save_csv(benchmark, benchmark_file)
        print("="*60)
        print("MOTION EXTRACTION BENCHMARK")
        print("="*60)
        print(benchmark)
        print()
        print(f"Benchmark saved :\n{benchmark_file}")
        print("="*60)
        return benchmark


if __name__ == "__main__":
    pipeline = MotionPipeline()
    pipeline.run()
