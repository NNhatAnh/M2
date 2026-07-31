import numpy as np
import time
import pandas as pd
from config import *
from utils import *
from metrics import *
from visualization import *
from report import *


class OutlierPipeline:
    def __init__(self):
        self.datasets = {}
        self.cleaned = {}
        self.summary = []

    # =====================================================
    # Load interpolated data
    # =====================================================
    def load_dataset(self):
        print_title("Load Interpolated Dataset")
        for name in DATASETS.keys():
            folder = create_output_folder("interpolation", name)
            file = folder / "interpolated.csv"
            df = pd.read_csv(file)
            self.datasets[name] = df
            print(f"{name} : {len(df)} samples")
        print()

    # =====================================================
    # Z-score Detection
    # =====================================================
    def modified_zscore_detect(self, signal, threshold=3.5):
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))

        if mad == 0:
            return np.zeros(len(signal), dtype=bool)

        modified_z = 0.6745 * (signal - median) / mad

        return np.abs(modified_z) > threshold

    # =====================================================
    # IQR Detection
    # =====================================================
    def iqr_detect(self, signal, factor=1.5):
        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1
        lower = q1 - factor*iqr
        upper = q3 + factor*iqr
        return ((signal < lower) | (signal > upper))

    # =====================================================
    # Detect Outlier
    # =====================================================
    def detect(self):
        print_title("Detect Outlier")
        for name, df in self.datasets.items():
            print(f"Dataset : {name}")
            result = {}
            for feature in FEATURE_COLUMNS:
                signal = df[feature].values
                mask_mz = self.modified_zscore_detect(signal)
                mask_iqr = self.iqr_detect(signal)
                mask = mask_mz & mask_iqr
                result[feature] = mask
                print(f"{feature:<15}" f" Outlier : {mask.sum()}")
            self.cleaned[name] = {
                "data": df,
                "mask": result
            }
            print()

    # =====================================================
    # Remove Outlier
    # =====================================================
    def remove(self):
        print_title("Remove Outlier")
        for name in self.datasets.keys():
            df = self.cleaned[name]["data"].copy()
            mask = self.cleaned[name]["mask"]
            removed = {}
            for feature in FEATURE_COLUMNS:
                signal = df[feature].values.astype(float)
                outlier_mask = self.remove_long_gap(mask[feature])
                signal[outlier_mask] = np.nan

                # Linear interpolation
                signal = pd.Series(signal).interpolate(
                    method="linear",
                    limit_direction="both"
                )

                # Fill remaining NaN
                signal = signal.bfill()
                signal = signal.ffill()
                df[feature] = signal.values
                removed[feature] = int(outlier_mask.sum())
            self.cleaned[name]["clean_df"] = df
            self.cleaned[name]["removed"] = removed
            output = create_output_folder("outlier", name)
            save_csv(df, output / "cleaned.csv")
            print(f"{name} finished.")
        print()

    def remove_long_gap(self, mask, max_gap=5):
        mask = mask.copy()
        start = None

        for i in range(len(mask)):
            if mask[i] and start is None:
                start = i
            elif (not mask[i]) and start is not None:
                if i - start > max_gap:
                    mask[start:i] = False
                start = None

        if start is not None:
            if len(mask)-start > max_gap:
                mask[start:] = False

        return mask

    # =====================================================
    # Evaluation
    # =====================================================
    def evaluate(self):
        print_title("Evaluate Outlier Removal")
        for name in self.datasets.keys():
            raw_df = self.datasets[name]
            clean_df = self.cleaned[name]["clean_df"]

            removed = self.cleaned[name]["removed"]
            report = Report()
            report.title(f"{name.upper()} OUTLIER REPORT")
            summary = []

            for feature in FEATURE_COLUMNS:
                raw = raw_df[feature].values
                clean = clean_df[feature].values
                removed_ratio = removed[feature] / len(raw) * 100
                corr = np.corrcoef(raw, clean)[0, 1]
                energy = (np.sum(clean**2) / np.sum(raw**2)) * 100

                correction = np.mean(
                    np.abs(raw-clean)
                )
                feature_mse = mse(raw, clean)

                report.add(f"{feature} Removed", removed[feature])
                report.add(f"{feature} Correlation", round(corr, 6))
                report.add(f"{feature} Energy", round(energy, 6))
                report.add(f"{feature} Correction", round(correction, 6))
                report.add(f"{feature} Removed Ratio", round(removed_ratio, 6))
                report.blank()

                summary.append({
                    "Feature": feature,
                    "Removed": removed[feature],
                    "Correlation": round(corr, 6),
                    "Energy": round(energy, 6),
                    "Correction": round(correction, 6),
                    "Removed Ratio": round(removed_ratio, 6),
                    "MSE": feature_mse
                })
            output = create_output_folder("outlier", name)
            report.save(output / "report.txt")

            save_summary(summary, output / "summary.csv")
            print(f"{name} evaluated.")
        print()

    # =====================================================
    # Visualization
    # =====================================================
    def plot(self):
        print_title("Visualization")
        for name in self.datasets.keys():
            output = create_output_folder(
                "outlier",
                name
            )
            raw_df = self.datasets[name]
            clean_df = self.cleaned[name]["clean_df"]

            # ==============================================
            # Plot every feature
            # ==============================================
            for feature in FEATURE_COLUMNS:
                plot_compare(
                    raw_df["time_s"],
                    raw_df[feature],
                    clean_df[feature],
                    "Before",
                    "After",
                    feature,
                    output / f"{feature}_compare.png"
                )
                plot_signal(
                    clean_df["time_s"],
                    clean_df[feature],
                    f"{feature} (Clean)",
                    "Time (s)",
                    feature,
                    output / f"{feature}_clean.png"
                )
            print(f"{name} finished.")
        print()

    # =====================================================
    # Export Summary
    # =====================================================
    def export(self):
        summary = []
        for name in self.cleaned.keys():
            removed = self.cleaned[name]["removed"]
            total_removed = 0
            for feature in FEATURE_COLUMNS:
                total_removed += removed[feature]
            summary.append({
                "Dataset": name,
                "Total Removed": total_removed
            })
        save_summary(
            summary,
            OUTLIER_DIR / "dataset_summary.csv"
        )

        print("Summary exported.\n")

    # =====================================================
    # Run Pipeline
    # =====================================================
    def run(self):
        start = time.perf_counter()
        self.load_dataset()
        self.detect()
        self.remove()
        self.evaluate()
        self.plot()
        self.export()
        runtime = time.perf_counter() - start
        print(f"Outlier Removal Pipeline completed in {runtime:.2f} seconds.\n")


if __name__ == "__main__":
    pipeline = OutlierPipeline()
    pipeline.run()
