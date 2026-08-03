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
# CONFIG
# ==========================================================
METHODS = [
    "butterworth",
    "median",
    "moving_average"
]


# ==========================================================
# Statistical Features
# ==========================================================
def statistical_features(signal):
    feature = {}
    signal = np.asarray(signal)

    feature["Mean"] = np.mean(signal)
    feature["Median"] = np.median(signal)
    feature["STD"] = np.std(signal)
    feature["Variance"] = np.var(signal)
    feature["RMS"] = np.sqrt(np.mean(signal**2))
    feature["IQR"] = iqr(signal)
    feature["Skewness"] = skew(signal)
    feature["Kurtosis"] = kurtosis(signal)
    feature["CoeffVar"] = (np.std(signal) / (np.mean(signal) + 1e-12))

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

    # ==========================================================
    # Temporal Features
    # ==========================================================
    def temporal_features(self, signal):
        feature = {}
        signal = np.asarray(signal)

        # --------------------------------------
        # Signal Energy
        # --------------------------------------
        feature["Energy"] = np.sum(signal**2)

        # --------------------------------------
        # Signal Power
        # --------------------------------------
        feature["Power"] = np.mean(signal**2)

        # --------------------------------------
        # Zero Crossing Rate
        # --------------------------------------
        center = np.mean(signal)

        feature["ZeroCrossingRate"] = np.sum(
            np.diff(signal > center)) / len(signal)

        # --------------------------------------
        # Peak Detection
        # --------------------------------------
        peaks, properties = find_peaks(
            signal,
            height=np.mean(signal),
            distance=TARGET_FS * 0.5
        )
        feature["PeakCount"] = len(peaks)
        if len(peaks):
            feature["PeakHeight"] = np.mean(
                properties["peak_heights"]
            )
            prominence = peak_prominences(
                signal,
                peaks
            )[0]
            feature["PeakProminence"] = np.mean(
                prominence
            )
            widths = peak_widths(
                signal,
                peaks,
                rel_height=0.5
            )[0]
            feature["PeakWidth"] = np.mean(
                widths
            )
        else:
            feature["PeakHeight"] = 0
            feature["PeakProminence"] = 0
            feature["PeakWidth"] = 0

        # --------------------------------------
        # Activity Ratio
        # --------------------------------------
        binary = (
            signal >
            np.mean(signal)
        ).astype(np.uint8)
        feature["ActivityRatio"] = (
            np.mean(binary)
            * 100
        )
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

        # --------------------------------------
        # FFT Energy
        # --------------------------------------
        feature["FFTEnergy"] = np.sum(
            spectrum**2
        )

        # --------------------------------------
        # Dominant Frequency
        # --------------------------------------
        feature["DominantFrequency"] = freq[
            np.argmax(spectrum)
        ]

        # --------------------------------------
        # Spectral Centroid
        # --------------------------------------
        feature["SpectralCentroid"] = (
            np.sum(freq*spectrum) / (np.sum(spectrum) + 1e-12)
        )

        # --------------------------------------
        # Spectral Bandwidth
        # --------------------------------------
        centroid = feature["SpectralCentroid"]
        feature["Bandwidth"] = np.sqrt(
            np.sum(((freq-centroid)**2) * spectrum) /
            (np.sum(spectrum) + 1e-12)
        )

        # --------------------------------------
        # Spectral Entropy
        # --------------------------------------
        p = spectrum / (
            np.sum(spectrum)+1e-12
        )

        feature["SpectralEntropy"] = -np.sum(
            p*np.log2(p+1e-12)
        )

        # --------------------------------------
        # Spectral Flatness
        # --------------------------------------
        feature["SpectralFlatness"] = (
            np.exp(np.mean(np.log(spectrum+1e-12))) /
            (np.mean(spectrum) + 1e-12)
        )

        # --------------------------------------
        # Band Power
        # --------------------------------------
        cutoff = np.max(freq) * 0.25
        low = spectrum[freq < cutoff]
        high = spectrum[freq >= cutoff]

        feature["LowBandPower"] = np.sum(low**2)
        feature["HighBandPower"] = np.sum(high**2)

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
        for feature_name in FEATURE_COLUMNS:
            print(
                f"Extract : {feature_name}"
            )
            signal = df[feature_name].values
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
                "Method": method,
                "Dataset": dataset,
                "Signal": feature_name
            }
            result.update(stat)
            result.update(temp)
            result.update(freq)
            feature_table.append(
                result
            )
            feature_vector = {
                "Method": method,
                "Dataset": dataset,
                "Label": 0 if dataset == "static" else 1
            }

            for k,v in stat.items():
                feature_vector[
                    f"{feature_name}_{k}"] = v

            for k,v in temp.items():
                feature_vector[
                    f"{feature_name}_{k}"] = v

            for k,v in freq.items():
                feature_vector[
                    f"{feature_name}_{k}"] = v

            vector_df = pd.DataFrame([feature_vector])
            vector_df.to_csv(output_folder / "feature_vector.csv", index=False)

            # ----------------------------
            # Summary
            # ----------------------------
            summary.append({
               "Signal": feature_name,
               "Mean": stat["Mean"],
               "STD": stat["STD"],
               "RMS": stat["RMS"],
               "CoeffVar": stat["CoeffVar"],
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
        runtime = (
            time.perf_counter() - start
        )

        report.add(
            "Runtime(s)",
            round(
                runtime,
                4
            )
        )

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
            "Runtime(s)": runtime,
            "Mean STD": feature_df["STD"].mean(),
            "Mean RMS": feature_df["RMS"].mean(),
            "Mean Energy": feature_df["Energy"].mean(),
            "Mean Peak Count": feature_df["PeakCount"].mean(),
            "Mean Entropy": feature_df["SpectralEntropy"].mean(),
            "Mean FFT Energy": feature_df["FFTEnergy"].mean(),
            "Mean Peak Height": feature_df["PeakHeight"].mean(),
            "Mean Peak Prominence": feature_df["PeakProminence"].mean(),
            "Mean Activity": feature_df["ActivityRatio"].mean(),
            "Mean DominantFreq": feature_df["DominantFrequency"].mean(),
            "Mean Bandwidth": feature_df["Bandwidth"].mean()
        }

        self.summary.append(
            benchmark_row
        )
        print()
        print(f"Finished : {method} - {dataset}")
        print(f"Output : {output_folder}")
        print()

    # ======================================================
    # Run
    # ======================================================
    def run(self):
        print_title("MODULE 06 : FEATURE EXTRACTION")
        for method in METHODS:
            for dataset in DATASETS.keys():
                self.process(
                    method,
                    dataset
                )

        benchmark = pd.DataFrame(
            self.summary
        )
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
