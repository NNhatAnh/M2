"""
==========================================================
MODULE 06 - FEATURE EXTRACTION
==========================================================

Input
------
results/motion_extraction/

Output
------
results/feature/

==========================================================
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import iqr
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import peak_prominences
from scipy.fft import fft
from config import *
from utils import *
from report import *


# ==========================================================
# Statistical Features
# ==========================================================
def statistical_features(signal):
    feature = {}
    signal = np.asarray(signal)

    feature["Median"] = np.median(signal)
    feature["STD"] = np.std(signal)
    feature["RMS"] = np.sqrt(np.mean(signal**2))
    feature["IQR"] = iqr(signal)
    feature["Skewness"] = skew(signal)
    feature["Kurtosis"] = kurtosis(signal)

    return feature


# ==========================================================
# Feature Pipeline
# ==========================================================
class FeaturePipeline:
    def __init__(self):
        self.summary = []

    # ======================================================
    # Load
    # ======================================================
    def load_dataset(self, method, dataset):
        file = (MOTION_DIR / method / dataset / "motion_score.csv")

        if not file.exists():
            raise FileNotFoundError(file)
        return pd.read_csv(file)

    # ======================================================
    # Sliding Window
    # ======================================================
    def create_windows(self, df):
        windows = []
        N = len(df)
        start = 0

        while start+WINDOW_SIZE <= N:
            windows.append(df.iloc[start:start+WINDOW_SIZE].copy())
            start += STEP_SIZE
        return windows

    # ==========================================================
    # Temporal Features
    # ==========================================================
    def temporal_features(self, signal):
        feature = {}
        signal = np.asarray(signal)

        feature["Energy"] = np.sum(signal**2)
        peaks, properties = find_peaks(
            signal,
            height=np.mean(signal),
            distance=TARGET_FS * 0.5
        )
        feature["PeakCount"] = len(peaks)
        if len(peaks):
            feature["PeakHeight"] = np.mean(properties["peak_heights"])
            prominence = peak_prominences(
                signal,
                peaks
            )[0]
            feature["PeakProminence"] = np.mean(prominence)
        else:
            feature["PeakHeight"] = 0
            feature["PeakProminence"] = 0
        binary = (
            signal >
            np.mean(signal)
        ).astype(np.uint8)
        feature["ActivityRatio"] = (np.mean(binary) * 100)
        feature["MotionDuration"] = np.sum(binary)
        return feature

    # ==========================================================
    # Frequency Features
    # ==========================================================
    def frequency_features(self, signal):
        feature = {}
        signal = np.asarray(signal)

        signal = signal - np.mean(signal)
        N = len(signal)

        spectrum = np.abs(fft(signal))[:N//2]
        freq = np.fft.fftfreq(N, d=1 / TARGET_FS)[:N//2]

        feature["FFTEnergy"] = np.sum(
            spectrum**2
        )

        feature["SpectralCentroid"] = (
            np.sum(freq*spectrum) / (np.sum(spectrum) + 1e-12)
        )

        centroid = feature["SpectralCentroid"]
        feature["Bandwidth"] = np.sqrt(
            np.sum(((freq-centroid)**2) * spectrum) /
            (np.sum(spectrum) + 1e-12)
        )

        p = spectrum / (
            np.sum(spectrum)+1e-12
        )

        feature["SpectralEntropy"] = -np.sum(
            p*np.log2(p+1e-12)
        )

        return feature

    # ======================================================
    # Process
    # ======================================================
    def process(self, method, dataset):
        print_title(f"{method.upper()} : {dataset.upper()}")

        df = self.load_dataset(
            method,
            dataset
        )

        windows = self.create_windows(df)
        print(f"Total windows: {len(windows)}")

        output_folder = (FEATURE_DIR / method / dataset)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        report = Report()
        report.title(
            f"{method.upper()} - {dataset.upper()}"
        )

        feature_table = []
        summary = []
        start = time.perf_counter()

        # ==================================================
        # Feature Extraction
        # ==================================================
        feature_vectors = []

        for window_id, window in enumerate(windows):
            print(f"Window {window_id+1}/{len(windows)}")
            feature_vector = {
                "Sample": f"{method}_{dataset}_{window_id:03d}",
                "Method": method,
                "Dataset": dataset,
                "Label": 0 if dataset == "static" else 1
            }
            for feature_name in FEATURE_COLUMNS:
                print(
                    f"Extract : {feature_name}"
                )
                signal = window[feature_name].values
                stat = statistical_features(
                    signal
                )
                temp = self.temporal_features(
                    signal
                )
                freq = self.frequency_features(
                    signal
                )
                result = {
                    "Sample": f"{method}_{dataset}_{window_id:03d}",
                    "Method": method,
                    "Dataset": dataset,
                    "Window": window_id,
                    "Signal": feature_name
                }
                result.update(stat)
                result.update(temp)
                result.update(freq)
                feature_table.append(
                    result
                )

                for k, v in stat.items():
                    feature_vector[
                        f"{feature_name}_{k}"] = v

                for k, v in temp.items():
                    feature_vector[
                        f"{feature_name}_{k}"] = v

                for k, v in freq.items():
                    feature_vector[
                        f"{feature_name}_{k}"] = v

                # ----------------------------
                # Summary
                # ----------------------------
                summary.append({
                    "Signal": feature_name,
                    "STD": stat["STD"],
                    "RMS": stat["RMS"],
                    "Energy": temp["Energy"],
                    "PeakCount": temp["PeakCount"],
                    "PeakHeight": temp["PeakHeight"],
                    "PeakProminence": temp["PeakProminence"],
                    "ActivityRatio": temp["ActivityRatio"],
                    "SpectralEntropy": freq["SpectralEntropy"],
                    "SpectralCentroid": freq["SpectralCentroid"],
                    "Bandwidth": freq["Bandwidth"],
                    "FFTEnergy": freq["FFTEnergy"]
                })

                # ----------------------------
                # Report
                # ----------------------------
                report.add(feature_name, "")
                for k, v in result.items():
                    if k in [
                        "Method",
                        "Dataset",
                        "Signal"
                    ]:
                        continue
                    if isinstance(v, float):
                        report.add(k, round(v, 4))
                    else:
                        report.add(k, v)
                report.blank()
            feature_vectors.append(feature_vector)
        vector_df = pd.DataFrame(feature_vectors)

        vector_df.to_csv(output_folder / "feature_vector.csv", index=False)

        # ==================================================
        # Save CSV
        # ==================================================
        feature_df = pd.DataFrame(
            feature_table
        )
        summary_df = pd.DataFrame(
            summary
        )
        feature_df.to_csv(output_folder / "features.csv", index=False)
        summary_df.to_csv(output_folder / "summary.csv", index=False)
        report.save(output_folder / "report.txt")

        # ==================================================
        # Visualization
        # ==================================================
        figure_folder = (
            output_folder
            /
            "figures"
        )

        figure_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for feature_name in FEATURE_COLUMNS:
            signal = df[feature_name].values
            spectrum = np.abs(
                fft(signal)
            )[:len(signal)//2]
            freq = np.fft.fftfreq(
                len(signal),
                d=1/TARGET_FS
            )[:len(signal)//2]
            plt.figure(figsize=(12, 8))

            # ------------------------------------------
            # Motion Score
            # ------------------------------------------
            plt.subplot(311)
            plt.plot(
                df["time_s"],
                signal,
                linewidth=1
            )
            plt.title(
                feature_name
            )
            plt.grid(alpha=0.3)

            # ------------------------------------------
            # Histogram
            # ------------------------------------------
            plt.subplot(312)
            plt.hist(
                signal,
                bins=30,
                edgecolor="black"
            )
            plt.title(
                "Histogram"
            )
            plt.grid(alpha=0.3)

            # ------------------------------------------
            # FFT
            # ------------------------------------------
            plt.subplot(313)
            plt.plot(
                freq,
                spectrum
            )
            plt.title(
                "FFT Spectrum"
            )
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                figure_folder
                /
                f"{feature_name}.png",
                dpi=300
            )
            plt.close()

        # ==================================================
        # Benchmark
        # ==================================================
        benchmark_row = {
            "Method": method,
            "Dataset": dataset,
            "Mean STD": feature_df["STD"].mean(),
            "Mean RMS": feature_df["RMS"].mean(),
            "Mean Energy": feature_df["Energy"].mean(),
            "Mean Peak Count": feature_df["PeakCount"].mean(),
            "Mean Entropy": feature_df["SpectralEntropy"].mean(),
            "Mean FFT Energy": feature_df["FFTEnergy"].mean(),
            "Mean Peak Height": feature_df["PeakHeight"].mean(),
            "Mean Peak Prominence": feature_df["PeakProminence"].mean(),
            "Mean Activity": feature_df["ActivityRatio"].mean(),
            "Mean Bandwidth": feature_df["Bandwidth"].mean()
        }

        self.summary.append(
            benchmark_row
        )
        print()
        print(f"Finished : {method} - {dataset}")
        print(f"Output : {output_folder}")
        print()

    # ==========================================================
    # Filter Comparison Visualization
    # ==========================================================
    def plot_filter_comparison(self, dataset):
        """
        Compare Butterworth, Median and Moving Average
        for the same recording.

        Rows:
            1. Time domain
            2. Frequency domain

        Columns:
            Butterworth
            Median
            Moving Average
        """

        print()
        print("=" * 60)
        print(f"FILTER COMPARISON : {dataset.upper()}")
        print("=" * 60)

        methods = [
            "butterworth",
            "median",
            "moving_average"
        ]

        # ------------------------------------------------------
        # Select one representative signal
        # ------------------------------------------------------

        plot_feature = "amp_mean"

        data = {}

        for method in methods:

            file = (
                MOTION_DIR /
                method /
                dataset /
                "motion_score.csv"
            )

            if not file.exists():
                print(f"Missing : {file}")
                continue

            df = pd.read_csv(file)

            if plot_feature not in df.columns:
                print(
                    f"{plot_feature} not found in {file}"
                )
                continue

            time_data = df[plot_feature].values

            # Remove DC component before FFT
            x = time_data - np.mean(time_data)

            N = len(x)

            spectrum = np.abs(
                fft(x)
            )[:N // 2]

            freq = np.fft.fftfreq(
                N,
                d=1 / TARGET_FS
            )[:N // 2]

            data[method] = {
                "time": df["time_s"].values,
                "signal": time_data,
                "freq": freq,
                "spectrum": spectrum
            }

        if len(data) == 0:
            print("No data available for comparison.")
            return

        # ------------------------------------------------------
        # Create figure
        # ------------------------------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 9)
        )

        method_titles = {
            "butterworth": "Butterworth",
            "median": "Median",
            "moving_average": "Moving Average"
        }

        # ------------------------------------------------------
        # Plot
        # ------------------------------------------------------

        for col, method in enumerate(methods):

            if method not in data:
                axes[0, col].set_visible(False)
                axes[1, col].set_visible(False)
                continue

            d = data[method]

            # ----------------------------------------------
            # Time domain
            # ----------------------------------------------

            axes[0, col].plot(
                d["time"],
                d["signal"],
                linewidth=0.8
            )

            axes[0, col].set_title(
                method_titles[method]
            )

            axes[0, col].set_xlabel(
                "Time (s)"
            )

            axes[0, col].set_ylabel(
                "Amplitude"
            )

            axes[0, col].grid(
                alpha=0.3
            )

            # ----------------------------------------------
            # Frequency domain
            # ----------------------------------------------

            axes[1, col].plot(
                d["freq"],
                d["spectrum"],
                linewidth=0.8
            )

            axes[1, col].set_xlabel(
                "Frequency (Hz)"
            )

            axes[1, col].set_ylabel(
                "Magnitude"
            )

            axes[1, col].grid(
                alpha=0.3
            )

        # ------------------------------------------------------
        # Row titles
        # ------------------------------------------------------

        fig.text(
            0.02,
            0.73,
            "Time Domain",
            rotation=90,
            va="center",
            fontsize=12,
            fontweight="bold"
        )

        fig.text(
            0.02,
            0.30,
            "Frequency Domain",
            rotation=90,
            va="center",
            fontsize=12,
            fontweight="bold"
        )

        fig.suptitle(
            f"Filter Comparison - {dataset.upper()} - {plot_feature}",
            fontsize=15
        )

        plt.tight_layout(
            rect=[0.04, 0.03, 1, 0.95]
        )

        # ------------------------------------------------------
        # Save
        # ------------------------------------------------------

        comparison_folder = (
            FEATURE_DIR /
            "filter_comparison"
        )

        comparison_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            comparison_folder /
            f"{dataset}_comparison.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Saved : {output_file}"
        )

        # ------------------------------------------------------
        # Display
        # ------------------------------------------------------

        plt.show()
        plt.close()

    # ======================================================
    # Run
    # ======================================================
    def run(self):

        print_title(
            "MODULE 06 : FEATURE EXTRACTION"
        )

        # ======================================================
        # Feature Extraction
        # ======================================================

        for method in METHODS:

            for dataset in DATASETS.keys():

                self.process(
                    method,
                    dataset
                )

        # ======================================================
        # Filter Comparison Visualization
        # ======================================================

        for dataset in DATASETS.keys():

            self.plot_filter_comparison(
                dataset
            )

        # ======================================================
        # Benchmark
        # ======================================================

        benchmark = pd.DataFrame(
            self.summary
        )

        BENCHMARK_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        benchmark.to_csv(
            BENCHMARK_DIR /
            "feature_benchmark.csv",
            index=False
        )

        print("=" * 60)
        print("FEATURE EXTRACTION BENCHMARK")
        print("=" * 60)
        print(benchmark)

        print("=" * 60)

        return benchmark
        BENCHMARK_DIR.mkdir(
            parents=True,
            exist_ok=True
        )
        benchmark.to_csv(BENCHMARK_DIR / "feature_benchmark.csv", index=False)
        print("="*60)
        print("FEATURE EXTRACTION BENCHMARK")
        print("="*60)
        print(benchmark)

        print("="*60)
        return benchmark


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
