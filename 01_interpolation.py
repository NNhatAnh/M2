import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from scipy.interpolate import interp1d
from config import *
from utils import *
from metrics import *
from visualization import *
from report import *


class InterpolationPipeline:
    def __init__(self):
        self.datasets = {}
        self.results = {}
        self.summary = []

    # ======================================================
    # Load Dataset
    # ======================================================
    def load_dataset(self):
        print_title("Loading Dataset")
        for name, path in DATASETS.items():
            print(f"Loading : {name}")
            df = read_csv(path)
            df = convert_time(df)
            self.datasets[name] = df
            print(f"Samples : {len(df)}")
            print(f"Duration : {df['time_s'].iloc[-1]:.3f} s")
            print()

    # ======================================================
    # Analyze Dataset
    # ======================================================
    def analyze_dataset(self):
        print_title("Dataset Analysis")
        for name, df in self.datasets.items():
            time = df["time_s"].values
            dt = packet_interval(time)
            fs = sampling_frequency(time)
            jitter = packet_jitter(time)
            psi_value = psi(time)

            print("-----------------------------------")
            print(f"Dataset : {name}")
            print(f"Average Fs : {fs:.3f} Hz")
            print(f"Packet Jitter : {jitter:.6f} s")
            print(f"PSI : {psi_value:.4f}")
            print()

            report = Report()
            report.title(f"{name.upper()} DATASET")
            report.add("Samples", len(df))
            report.add("Duration (s)", round(time[-1], 3))
            report.add("Sampling Frequency", round(fs, 3))
            report.add("Packet Jitter", round(jitter, 6))
            report.add("Packet Stability Index", round(psi_value, 4))

            output = create_output_folder("interpolation", name)
            report.save(
                output / "dataset_report.txt"
            )

            self.summary.append({
                "Dataset": name,
                "Samples": len(df),
                "Duration": time[-1],
                "Fs": fs,
                "Jitter": jitter,
                "PSI": psi_value
            })
            self.results[name] = {
                "time": time,
                "dt": dt,
                "fs": fs,
                "jitter": jitter,
                "psi": psi_value
            }

    # ======================================================
    # Linear Interpolation
    # ======================================================
    def interpolate(self):
        print_title("Linear Interpolation")
        for name, df in self.datasets.items():
            print(f"Processing : {name}")
            time = df["time_s"].values
            dt = 1 / TARGET_FS
            new_time = np.arange(time[0], time[-1], dt)
            interp_df = pd.DataFrame()
            interp_df["time_s"] = new_time

            # ---------------------------------------------
            # Interpolate every feature
            # ---------------------------------------------
            for feature in FEATURE_COLUMNS:
                f = interp1d(
                    time,
                    df[feature].values,
                    kind=INTERPOLATION_METHOD,
                    fill_value="extrapolate"
                )
                interp_df[feature] = f(new_time)

            # Copy label
            if LABEL_COLUMN in df.columns:
                interp_df[LABEL_COLUMN] = df[LABEL_COLUMN].iloc[0]

            # Save result
            output = create_output_folder("interpolation", name)
            save_csv(interp_df, output / "interpolated.csv")

            # Save into memory
            self.results[name]["interp"] = interp_df
            self.results[name]["new_time"] = new_time
            print(f"Original Samples : {len(df)}")
            print(f"Interpolated Samples : {len(interp_df)}")
            print()

    # ======================================================
    # Evaluate Interpolation
    # ======================================================
    def evaluate(self):
        print_title("Interpolation Evaluation")
        for name, df in self.datasets.items():
            print(f"Evaluating : {name}")
            interp_df = self.results[name]["interp"]
            report = Report()
            report.title(f"{name.upper()} INTERPOLATION")
            metrics_result = {}

            # ---------------------------------------------
            # Evaluate every feature
            # ---------------------------------------------
            for feature in FEATURE_COLUMNS:
                raw_time = df["time_s"].values
                raw_signal = df[feature].values
                interp_time = interp_df["time_s"].values
                interp_signal = interp_df[feature].values

                # Map interpolated signal back to original timestamps
                f = interp1d(
                    interp_time,
                    interp_signal,
                    kind="linear",
                    fill_value="extrapolate"
                )
                reconstructed = f(raw_time)
                feature_rmse = rmse(raw_signal, reconstructed)
                feature_mae = mae(raw_signal, reconstructed)
                feature_mse = mse(raw_signal, reconstructed)
                metrics_result[feature] = {
                    "RMSE": feature_rmse,
                    "MAE": feature_mae,
                    "MSE": feature_mse
                }
                report.add(f"{feature} RMSE", round(feature_rmse, 6))
                report.add(f"{feature} MAE", round(feature_mae, 6))

            # Packet metrics
            report.blank()
            report.add("Sampling Frequency", round(
                self.results[name]["fs"], 3))
            report.add("Packet Jitter", round(self.results[name]["jitter"], 6))
            report.add("PSI", round(self.results[name]["psi"], 4))
            output = create_output_folder("interpolation", name)
            report.save(output / "evaluation_report.txt")
            summary = []
            for feature in FEATURE_COLUMNS:
                summary.append({
                    "Feature": feature,
                    "RMSE": metrics_result[feature]["RMSE"],
                    "MAE": metrics_result[feature]["MAE"],
                    "MSE": metrics_result[feature]["MSE"]
                })
            save_summary(summary, output / "summary.csv")
            self.results[name]["metrics"] = metrics_result
            print("Done")
            print()

    # ======================================================
    # Visualization
    # ======================================================
    def plot(self):
        print_title("Visualization")
        for name in self.datasets.keys():
            output = create_output_folder("interpolation", name)

            raw_df = self.datasets[name]
            interp_df = self.results[name]["interp"]

            # ==========================================
            # Plot every feature
            # ==========================================
            for feature in FEATURE_COLUMNS:
                # Raw signal
                plot_signal(
                    raw_df["time_s"],
                    raw_df[feature],
                    f"{feature} Raw Signal",
                    "Time (s)",
                    feature,
                    output / f"{feature}_raw.png"
                )

                # Interpolated signal
                plot_signal(
                    interp_df["time_s"],
                    interp_df[feature],
                    f"{feature} Interpolated",
                    "Time (s)",
                    feature,
                    output / f"{feature}_interp.png"
                )

                # Comparison
                plot_compare(
                    raw_df["time_s"],
                    raw_df[feature],
                    np.interp(raw_df["time_s"],
                              interp_df["time_s"], interp_df[feature]),
                    "Raw",
                    "Interpolated",
                    feature,
                    output / f"{feature}_compare.png"
                )

            # ==========================================
            # Packet Interval Histogram
            # ==========================================
            plot_histogram(
                self.results[name]["dt"],
                bins=30,
                title="Packet Interval",
                save_path=output / "packet_interval.png"
            )
            print(f"{name} figure saved.")
        print()

    # ======================================================
    # Export Summary
    # ======================================================
    def export(self):
        save_summary(self.summary, INTERP_DIR / "dataset_summary.csv")
        print("Summary exported.")

    # ======================================================
    # Run Pipeline
    # ======================================================
    def run(self):
        self.load_dataset()
        self.analyze_dataset()
        self.interpolate()
        self.evaluate()
        self.plot()
        self.export()


if __name__ == "__main__":
    pipeline = InterpolationPipeline()
    pipeline.run()
