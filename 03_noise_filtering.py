import numpy as np
import pandas as pd
import time
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import medfilt
from config import *
from utils import *
from metrics import *
from visualization import *
from report import *


class FilteringPipeline:
    def __init__(self):
        self.datasets = {}
        self.results = {}

    # =====================================================
    # Load Dataset
    # =====================================================
    def load_dataset(self):
        print_title("Load Clean Dataset")
        for name in DATASETS.keys():
            folder = create_output_folder(
                "outlier",
                name
            )
            file = folder / "cleaned.csv"
            df = pd.read_csv(file)
            self.datasets[name] = df
            print(
                f"{name} : {len(df)} samples"
            )
        print()

    # =====================================================
    # Butterworth
    # =====================================================
    def butterworth_filter(self, signal, cutoff=2, order=4):
        fs = TARGET_FS
        nyquist = fs / 2
        wn = cutoff / nyquist
        b, a = butter(
            order,
            wn,
            btype="low"
        )
        return filtfilt(b, a, signal)

    # =====================================================
    # Moving Average
    # =====================================================
    def moving_average(self, signal, window=5):
        kernel = np.ones(window)
        kernel /= window
        return np.convolve(
            signal,
            kernel,
            mode="same"
        )

    # =====================================================
    # Median Filter
    # =====================================================
    def median_filter(self, signal, kernel=5):
        return medfilt(
            signal,
            kernel_size=kernel
        )

    # =====================================================
    # Apply Filter
    # =====================================================
    def process(self):
        print_title("Noise Filtering")
        methods = {
            "butterworth": self.butterworth_filter,
            "moving_average": self.moving_average,
            "median": self.median_filter
        }

        for method_name, method in methods.items():
            print(f"Method : {method_name}")
            self.results[method_name] = {}

            for dataset_name, df in self.datasets.items():
                filtered = df.copy()

                for feature in FEATURE_COLUMNS:
                    filtered[feature] = method(df[feature].values)

                self.results[method_name][dataset_name] = filtered
            print("Done")
        print()

    # =====================================================
    # Evaluate Filtering
    # =====================================================
    def evaluate(self):
        print_title("Evaluate Filtering")
        benchmark = []
        for method_name in self.results.keys():
            print(f"Method : {method_name}")
            for dataset_name in self.results[method_name]:
                raw_df = self.datasets[dataset_name]
                filt_df = self.results[method_name][dataset_name]
                output = RESULT_DIR / "filtering" / method_name / dataset_name
                output.mkdir(parents=True, exist_ok=True)
                report = Report()
                report.title(
                    f"{method_name.upper()} - {dataset_name.upper()}"
                )
                summary = []
                for feature in FEATURE_COLUMNS:
                    raw = raw_df[feature].values.astype(float)
                    filt = filt_df[feature].values.astype(float)

                    # ---------------------------------------
                    # Basic
                    # ---------------------------------------
                    rmse_value = rmse(raw, filt)
                    mae_value = mae(raw, filt)

                    # ---------------------------------------
                    # Variance Reduction
                    # ---------------------------------------
                    raw_var = np.var(raw)
                    filt_var = np.var(filt)
                    if raw_var == 0:
                        var_reduction = 0
                    else:
                        var_reduction = ((raw_var - filt_var) / raw_var) * 100

                    # ---------------------------------------
                    # STD Reduction
                    # ---------------------------------------
                    raw_std = np.std(raw)
                    filt_std = np.std(filt)
                    if raw_std == 0:
                        std_reduction = 0
                    else:
                        std_reduction = ((raw_std - filt_std) / raw_std) * 100

                    # ---------------------------------------
                    # Correlation
                    # ---------------------------------------
                    corr = np.corrcoef(raw, filt)[0, 1]
                    if np.isnan(corr):
                        corr = 0

                    # ---------------------------------------
                    # Energy Preservation
                    # ---------------------------------------
                    raw_energy = np.sum(raw ** 2)
                    filt_energy = np.sum(filt ** 2)
                    if raw_energy == 0:
                        energy = 0
                    else:
                        energy = (filt_energy / raw_energy) * 100

                    # ---------------------------------------
                    # Smoothness
                    # ---------------------------------------
                    raw_diff = np.diff(raw)
                    filt_diff = np.diff(filt)
                    smooth_before = np.std(raw_diff)
                    smooth_after = np.std(filt_diff)
                    if smooth_before == 0:
                        smooth_gain = 0
                    else:
                        smooth_gain = ((smooth_before - smooth_after) / smooth_before) * 100

                    # ---------------------------------------
                    # Runtime
                    # ---------------------------------------
                    start = time.perf_counter()
                    runtime = (time.perf_counter() - start) * 1000

                    # ---------------------------------------
                    # Report
                    # ---------------------------------------
                    report.add(
                        f"{feature} Variance Reduction (%)",
                        round(var_reduction, 3)
                    )
                    report.add(
                        f"{feature} STD Reduction (%)",
                        round(std_reduction, 3)
                    )
                    report.add(
                        f"{feature} Correlation",
                        round(corr, 6)
                    )
                    report.add(
                        f"{feature} Energy Preservation (%)",
                        round(energy, 3)
                    )
                    report.add(
                        f"{feature} Smoothness Gain (%)",
                        round(smooth_gain, 3)
                    )
                    report.add(
                        f"{feature} Runtime (ms)",
                        round(runtime, 4)
                    )
                    report.blank()
                    summary.append({
                        "Feature": feature,
                        "Variance Reduction (%)": var_reduction,
                        "STD Reduction (%)": std_reduction,
                        "Correlation": corr,
                        "Energy Preservation (%)": energy,
                        "Smoothness Gain (%)": smooth_gain,
                        "Runtime (ms)": runtime,
                        "RMSE": rmse_value,
                        "MAE": mae_value
                    })
                report.save(
                    output / "report.txt"
                )
                save_summary(
                    summary,
                    output / "summary.csv"
                )
                benchmark.append({
                    "Method": method_name,
                    "Dataset": dataset_name,
                    "Variance Reduction":
                        np.mean([
                            x["Variance Reduction (%)"]
                            for x in summary
                        ]),
                    "STD Reduction":
                        np.mean([
                            x["STD Reduction (%)"]
                            for x in summary
                        ]),
                    "Correlation":
                        np.mean([
                            x["Correlation"]
                            for x in summary
                        ]),
                    "Energy Preservation":
                        np.mean([
                            x["Energy Preservation (%)"]
                            for x in summary
                        ]),
                    "Smoothness Gain":
                        np.mean([
                            x["Smoothness Gain (%)"]
                            for x in summary
                        ]),
                    "Runtime (ms)":
                        np.mean([
                            x["Runtime (ms)"]
                            for x in summary
                        ])
                })
        self.benchmark = benchmark
        print()

    # =====================================================
    # Plot
    # =====================================================

    def plot(self):
        print_title("Visualization")

        for method_name in self.results.keys():
            for dataset_name in self.results[method_name]:
                output = RESULT_DIR / "filtering" / method_name / dataset_name
                output.mkdir(parents=True, exist_ok=True)

                raw_df = self.datasets[dataset_name]
                filter_df = self.results[method_name][dataset_name]

                for feature in FEATURE_COLUMNS:
                    plot_compare(
                        raw_df["time_s"],
                        raw_df[feature],
                        filter_df[feature],
                        "Before",
                        "After",
                        feature,
                        output / f"{feature}.png"
                    )
        print()

    # =====================================================
    # Export
    # =====================================================
    def export(self):
        print_title("Export")
        benchmark_df = pd.DataFrame(self.benchmark)
        benchmark_df.to_csv(
            RESULT_DIR /
            "filtering" /
            "benchmark.csv",
            index=False
        )

        for method_name in self.results.keys():
            for dataset_name in self.results[method_name]:
                output = RESULT_DIR / "filtering" / method_name / dataset_name

                output.mkdir(
                    parents=True,
                    exist_ok=True
                )

                self.results[method_name][dataset_name].to_csv(
                    output / "filtered.csv",
                    index=False

                )
        print("Done\n")

    # =====================================================
    # Run Pipeline
    # =====================================================
    def run(self):
        self.load_dataset()
        self.process()
        self.evaluate()
        self.plot()
        self.export()


if __name__ == "__main__":
    pipeline = FilteringPipeline()
    pipeline.run()
