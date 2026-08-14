"""
==========================================================
MODULE 04 - BACKGROUND ESTIMATION
==========================================================
Input:
    results/filtering/<dataset>/filtered.csv

Output:
    results/background/<dataset>/
        background.csv
        motion.csv
        report.txt
        summary.csv
        figures/
==========================================================
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from config import *
from utils import *
from metrics import *
from visualization import *
from report import *


# ==========================================================
# EMA Background
# ==========================================================
def ema_background(signal, alpha=0.05):
    signal = np.asarray(signal)

    background = np.zeros_like(signal)

    background[0] = signal[0]

    for i in range(1, len(signal)):
        background[i] = (
            alpha * signal[i]
            + (1 - alpha) * background[i - 1]
        )

    return background


# ==========================================================
# Background Pipeline
# ==========================================================
class BackgroundPipeline:
    def __init__(self):
        self.module_name = "background"
        self.summary = []

    # ======================================================
    # Load Dataset
    # ======================================================
    def load_dataset(self, method, dataset_name):
        input_file = (
            FILTER_DIR / method / dataset_name / "filtered.csv" 
        )
        if not input_file.exists():
            raise FileNotFoundError(
                f"Cannot find file:\n{input_file}"
            )
        df = pd.read_csv(input_file)
        return df

    # ======================================================
    # Process
    # ======================================================
    def process(self, method, dataset_name):
        print_title(f"BACKGROUND ESTIMATION : {dataset_name.upper()}")
        df = self.  load_dataset(method, dataset_name)

        output_folder = (BACKGROUND_DIR / method / dataset_name)

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        background_df = pd.DataFrame()
        motion_df = pd.DataFrame()
        background_df["time_s"] = df["time_s"]
        motion_df["time_s"] = df["time_s"]
        if LABEL_COLUMN in df.columns:
            background_df[LABEL_COLUMN] = df[LABEL_COLUMN]
            motion_df[LABEL_COLUMN] = df[LABEL_COLUMN]
        report = Report()
        report.title(
            f"Background Estimation ({dataset_name})"
        )
        report.add("Samples", len(df))
        report.add("Duration(s)", round(df["time_s"].iloc[-1], 2))
        report.blank()
        start_time = time.perf_counter()
        metrics_result = []

        # ==================================================
        # Process every feature
        # ==================================================
        for feature in FEATURE_COLUMNS:
            print(f"Processing : {feature}")
            signal = df[feature].values

            background = ema_background(
                signal,
                alpha=0.05
            )

            motion = signal - background
            background_df[feature] = background
            motion_df[feature] = motion
            background_std = std(background)
            motion_std = std(motion)
            background_energy = energy(background)
            motion_energy = energy(motion)

            correlation = np.corrcoef(
                signal,
                background
            )[0, 1]

            metrics_result.append({
                "Feature": feature,
                "Background STD":
                    background_std,
                "Motion STD":
                    motion_std,
                "Background Energy":
                    background_energy,
                "Motion Energy":
                    motion_energy,
                "Correlation":
                    correlation
            })
            report.add(feature, "")
            report.add("Background STD", round(background_std, 6))
            report.add("Motion STD", round(motion_std, 6))
            report.add("Background Energy", round(background_energy, 2))
            report.add("Motion Energy", round(motion_energy, 2))
            report.add("Correlation", round(correlation, 4))
            report.blank()

        # ==================================================
        # Save CSV
        # ==================================================
        background_path = output_folder / "background.csv"
        motion_path = output_folder / "motion.csv"

        save_csv(background_df, background_path)
        save_csv(motion_df, motion_path)

        # ==================================================
        # Save Summary
        # ==================================================
        summary_df = pd.DataFrame(metrics_result)
        summary_path = output_folder / "summary.csv"
        save_csv(summary_df, summary_path)

        # ==================================================
        # Save Report
        # ==================================================
        report_path = output_folder / "report.txt"
        report.save(report_path)

        # ==================================================
        # Plot
        # ==================================================
        figure_folder = output_folder / "figures"

        figure_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        x = df["time_s"]

        for feature in FEATURE_COLUMNS:

            # ----------------------------
            # Original vs Background
            # ----------------------------
            plot_compare(
                x=x,
                y1=df[feature],
                y2=background_df[feature],
                label1="Filtered",
                label2="Background",
                title=f"{feature} - Background Estimation",
                save_path=figure_folder /
                f"{feature}_background.png"
            )

            # ----------------------------
            # Motion Signal
            # ----------------------------
            plot_signal(
                x=x,
                y=motion_df[feature],
                title=f"{feature} - Motion",
                xlabel="Time (s)",
                ylabel=feature,
                save_path=figure_folder /
                f"{feature}_motion.png",
                color="red"
            )

        # ==================================================
        # Benchmark Summary
        # ==================================================
        self.summary.append({
            "Dataset": dataset_name,
            "Mean Background STD": summary_df["Background STD"].mean(),
            "Mean Motion STD": summary_df["Motion STD"].mean(),
            "Mean Correlation": summary_df["Correlation"].mean(),
            "Mean Motion Energy": summary_df["Motion Energy"].mean()
        })

        print(f"Finished : {dataset_name}")
        print(f"Output : {output_folder}")
        print()

    # ======================================================
    # Run Pipeline
    # ======================================================
    def run(self):
        print_title(
            "MODULE 04 : BACKGROUND ESTIMATION"
        )

        for method in METHODS:
            for dataset_name in DATASETS.keys():
                self.process(method, dataset_name)

        # ==============================================
        # Save Benchmark
        # ==============================================
        benchmark_folder = BENCHMARK_DIR
        benchmark_folder.mkdir(
            parents=True,
            exist_ok=True
        )
        benchmark_file = (
            benchmark_folder /
            "background_benchmark.csv"
        )
        benchmark_df = pd.DataFrame(
            self.summary
        )
        save_csv(
            benchmark_df,
            benchmark_file
        )

        # ==============================================
        # Print Benchmark
        # ==============================================
        print("=" * 60)
        print("BACKGROUND ESTIMATION BENCHMARK")
        print("=" * 60)
        print(benchmark_df)
        print()
        print(
            f"Benchmark saved : {benchmark_file}"
        )
        print("=" * 60)
        return benchmark_df


if __name__ == "__main__":
    pipeline = BackgroundPipeline()
    pipeline.run()
