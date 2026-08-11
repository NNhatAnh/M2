"""
==========================================================
MODULE 07 - CLASSIFICATION
==========================================================

Input
------
results/feature/

Output
------
results/classification/

==========================================================
"""

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from config import *
from utils import *
from report import *


# ==========================================================
# Classification Pipeline
# ==========================================================
class ClassificationPipeline:
    def __init__(self):
        self.summary = []

    # ======================================================
    # Load all feature vectors
    # ======================================================
    def load_dataset(self, method):
        rows = []
        for dataset in DATASETS.keys():
            file = (FEATURE_DIR / method / dataset / "feature_vector.csv")

            if not file.exists():
                raise FileNotFoundError(file)
            df = pd.read_csv(file)
            rows.append(df)

        dataset = pd.concat(rows, ignore_index=True)
        return dataset

    # ======================================================
    # Save merged dataset
    # ======================================================
    def save_dataset(
        self,
        method,
        dataset
    ):
        out_dir = (CLASSIFICATION_DIR / method)
        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        dataset.to_csv(out_dir / "dataset.csv", index=False)
        return out_dir

    # ======================================================
    # Prepare Data
    # ======================================================
    def prepare_data(self, dataset):
        y = dataset["Label"].astype(int)

        # ======================================================
        # Remove metadata
        # ======================================================
        X = dataset.drop(
            columns=[
                "Sample",
                "Label",
                "Method",
                "Dataset"
            ],
            errors="ignore"
        )

        # ======================================================
        # Selected features
        # ======================================================
        selected_columns = []

        for col in X.columns:

            for feature in SELECTED_FEATURES:

                if col.endswith(feature):
                    selected_columns.append(col)
                    break

        X = X[selected_columns]

        X = X.astype(np.float64)

        feature_names = X.columns.tolist()

        print()
        print("=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        print(
            f"Original Features : {len(dataset.columns) - 4}"
        )
        print(
            f"Selected Features : {len(feature_names)}"
        )
        print()

        return X, y, feature_names

    def create_blocked_folds(self, dataset):
        print()
        print("=" * 60)
        print("BLOCKED CROSS VALIDATION")
        print("=" * 60)
        print(
            f"Number of folds : {N_BLOCKED_FOLDS}"
        )
        print(
            f"Purge windows   : {PURGE_WINDOWS}"
        )
        print()

        class_blocks = {}

        # ======================================================
        # Divide each class into temporal blocks
        # ======================================================
        for label in sorted(dataset["Label"].unique()):

            class_indices = np.where(
                dataset["Label"].values == label
            )[0]

            if len(class_indices) < N_BLOCKED_FOLDS:
                raise ValueError(
                    f"Label {label} has only "
                    f"{len(class_indices)} samples."
                )

            blocks = np.array_split(
                class_indices,
                N_BLOCKED_FOLDS
            )

            class_blocks[label] = blocks

            print(
                f"Label {label}: "
                f"{len(class_indices)} samples -> "
                f"{[len(x) for x in blocks]}"
            )

        # ======================================================
        # Build folds
        # ======================================================
        folds = []

        for fold_id in range(N_BLOCKED_FOLDS):

            train_indices = []
            test_indices = []

            for label, blocks in class_blocks.items():

                test_block = blocks[fold_id]

                # ------------------------------------------------
                # TEST
                # ------------------------------------------------
                test_indices.extend(
                    test_block.tolist()
                )

                # ------------------------------------------------
                # TRAIN = all other blocks
                # ------------------------------------------------
                for block_id, block in enumerate(blocks):

                    if block_id == fold_id:
                        continue

                    train_indices.extend(
                        block.tolist()
                    )

                # ------------------------------------------------
                # PURGE around test block
                # ------------------------------------------------
                class_indices = np.where(
                    dataset["Label"].values == label
                )[0]

                test_start_pos = np.where(
                    class_indices == test_block[0]
                )[0][0]

                test_end_pos = np.where(
                    class_indices == test_block[-1]
                )[0][0]

                purge_start = max(
                    0,
                    test_start_pos - PURGE_WINDOWS
                )

                purge_end = min(
                    len(class_indices),
                    test_end_pos + 1 + PURGE_WINDOWS
                )

                purge_indices = set(
                    class_indices[
                        purge_start:purge_end
                    ].tolist()
                )

                train_indices = [
                    idx
                    for idx in train_indices
                    if idx not in purge_indices
                ]

            train_indices = np.array(
                sorted(set(train_indices)),
                dtype=int
            )

            test_indices = np.array(
                sorted(set(test_indices)),
                dtype=int
            )

            folds.append(
                (
                    train_indices,
                    test_indices
                )
            )

            print(
                f"Fold {fold_id + 1}: "
                f"Train={len(train_indices)}, "
                f"Test={len(test_indices)}"
            )

        return folds

    # ======================================================
    # Dataset Information
    # ======================================================
    def dataset_information(self, dataset):
        print()
        print("="*60)
        print("DATASET")
        print("="*60)
        print()
        print(dataset.shape)
        print()
        print(dataset["Label"].value_counts())
        print()
        print(dataset.describe())

    # ======================================================
    # Create Model
    # ======================================================
    def create_model(self, model_name):
        if model_name == "logistic":
            model = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            )
        elif model_name == "svm":
            model = CalibratedClassifierCV(SVC(
                kernel="rbf",
                random_state=RANDOM_STATE
            ), ensemble=False)
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_name == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=RANDOM_STATE
            )
        else:
            raise ValueError(
                model_name
            )
        return model

    # ======================================================
    # Create Output Folder
    # ======================================================
    def create_output_folder(self, out_dir, model_name):
        folder = (out_dir / model_name)

        folder.mkdir(parents=True, exist_ok=True)

        return folder

    # ======================================================
    # Save Prediction
    # ======================================================
    def save_prediction(self, folder, sample, y_true, y_pred, y_score=None):
        df = pd.DataFrame({
            "Sample": sample,
            "GroundTruth": y_true,
            "Prediction": y_pred
        })
        if y_score is not None:
            df["Probability"] = y_score
        df.to_csv(
            folder / "prediction.csv",
            index=False
        )

    # ======================================================
    # Save Model
    # ======================================================
    def save_model(self, folder, model):
        joblib.dump(model, folder / "model.pkl")

    # ======================================================
    # Print Model
    # ======================================================
    def print_model_information(self, model_name, model):
        print()
        print("="*60)
        print(model_name.upper())
        print("="*60)
        print(model)
        print()

    # ======================================================
    # Model List
    # ======================================================
    def get_models(self):
        models = {}
        for model_name in MODELS:
            models[model_name] = self.create_model(
                model_name
            )
        return models

    # ======================================================
    # Train One Model
    # ======================================================
    def train_one_model(self, model_name, data):
        X = data["X"]
        y = data["y"]

        dataset = data["dataset"]

        feature_names = data["feature_names"]
        folds = data["folds"]
        out_dir = data["out_dir"]

        # ======================================================
        # Output
        # ======================================================
        folder = self.create_output_folder(
            out_dir,
            model_name
        )

        print()
        print("=" * 60)
        print(
            f"MODEL : {model_name.upper()}"
        )
        print("=" * 60)

        fold_results = []

        all_true = []
        all_pred = []
        all_score = []
        all_samples = []

        feature_importances = []

        total_start = time.perf_counter()

        # ======================================================
        # FOLD LOOP
        # ======================================================
        for fold_id, (train_idx, test_idx) in enumerate(
            folds,
            start=1
        ):

            print()
            print(
                f"Fold {fold_id}/{len(folds)}"
            )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # ==================================================
            # SCALER
            # FIT ONLY ON TRAIN
            # ==================================================
            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_test_scaled = scaler.transform(
                X_test
            )

            # ==================================================
            # NEW MODEL FOR EACH FOLD
            # ==================================================
            model = self.create_model(
                model_name
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            # ==================================================
            # Prediction
            # ==================================================
            y_pred = model.predict(
                X_test_scaled
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                y_score = model.predict_proba(
                    X_test_scaled
                )[:, 1]

            else:

                y_score = None

            # ==================================================
            # Fold metrics
            # ==================================================
            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            if y_score is not None:
                auc = roc_auc_score(
                    y_test,
                    y_score
                )
            else:
                auc = np.nan

            overall = np.nanmean([
                accuracy,
                precision,
                recall,
                f1,
                auc
            ])

            fold_results.append({
                "Fold": fold_id,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": auc,
                "Overall Score": overall,
                "Train Samples": len(train_idx),
                "Test Samples": len(test_idx)
            })

            print(
                f"  Accuracy  : {accuracy:.4f}"
            )
            print(
                f"  Precision : {precision:.4f}"
            )
            print(
                f"  Recall    : {recall:.4f}"
            )
            print(
                f"  F1        : {f1:.4f}"
            )
            print(
                f"  ROC AUC   : {auc:.4f}"
            )

            # ==================================================
            # OOF predictions
            # ==================================================
            all_true.extend(
                y_test.tolist()
            )

            all_pred.extend(
                y_pred.tolist()
            )

            all_samples.extend(
                dataset.iloc[test_idx]["Sample"].tolist()
            )

            if y_score is not None:
                all_score.extend(
                    y_score.tolist()
                )

            # ==================================================
            # Random Forest importance
            # ==================================================
            if (
                model_name == "random_forest"
                and hasattr(
                    model,
                    "feature_importances_"
                )
            ):

                feature_importances.append(
                    model.feature_importances_
                )

        # ======================================================
        # Runtime
        # ======================================================
        runtime = (
            time.perf_counter()
            - total_start
        )

        fold_df = pd.DataFrame(
            fold_results
        )

        # ======================================================
        # Mean / STD
        # ======================================================
        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
            "Overall Score"
        ]

        means = fold_df[
            metric_columns
        ].mean()

        stds = fold_df[
            metric_columns
        ].std(
            ddof=1
        )

        # ======================================================
        # OOF arrays
        # ======================================================
        all_true = np.asarray(
            all_true
        )

        all_pred = np.asarray(
            all_pred
        )

        if all_score:
            all_score = np.asarray(
                all_score
            )
        else:
            all_score = None

        # ======================================================
        # Save fold results
        # ======================================================
        fold_df.to_csv(
            folder / "fold_results.csv",
            index=False
        )

        # ======================================================
        # Save OOF predictions
        # ======================================================
        self.save_prediction(
            folder,
            all_samples,
            all_true,
            all_pred,
            all_score
        )

        # ======================================================
        # Average RF feature importance
        # ======================================================
        if feature_importances:

            mean_importance = np.mean(
                feature_importances,
                axis=0
            )

            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": mean_importance
            })

            importance = importance.sort_values(
                "Importance",
                ascending=False
            )

            importance.to_csv(
                folder / "feature_importance.csv",
                index=False
            )

        # ======================================================
        # Result
        # ======================================================
        return {

            "Method":
                dataset["Method"].iloc[0],

            "Model":
                model_name,

            "Runtime":
                runtime,

            "Folder":
                folder,

            "FeatureNames":
                feature_names,

            "GroundTruth":
                all_true,

            "Prediction":
                all_pred,

            "Probability":
                all_score,

            "FoldResults":
                fold_df,

            "Accuracy":
                means["Accuracy"],

            "Precision":
                means["Precision"],

            "Recall":
                means["Recall"],

            "F1":
                means["F1 Score"],

            "AUC":
                means["ROC AUC"],

            "Overall Score":
                means["Overall Score"],

            "Accuracy STD":
                stds["Accuracy"],

            "Precision STD":
                stds["Precision"],

            "Recall STD":
                stds["Recall"],

            "F1 STD":
                stds["F1 Score"],

            "AUC STD":
                stds["ROC AUC"]
        }

    # ======================================================
    # Train All Models
    # ======================================================
    def train_all_models(self, data):
        results = []

        for model_name in MODELS:
            result = self.train_one_model(
                model_name,
                data
            )

            results.append(
                result
            )

        return results

    # ======================================================
    # Process
    # ======================================================
    def process(self, method):
        print()
        print("=" * 70)
        print(f"CLASSIFICATION : {method.upper()}")
        print("=" * 70)

        # ======================================================
        # Load
        # ======================================================
        dataset = self.load_dataset(method)
        out_dir = self.save_dataset(method, dataset)
        self.dataset_information(dataset)

        # ======================================================
        # Prepare
        # ======================================================
        X, y, feature_names = self.prepare_data(dataset)

        # ======================================================
        # Leakage-safe split
        # ======================================================
        folds = self.create_blocked_folds(dataset)

        # ======================================================
        # Data
        # ======================================================
        data = {
            "dataset": dataset,
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "folds": folds,
            "out_dir": out_dir
        }

        # ======================================================
        # Train
        # ======================================================
        results = self.train_all_models(data)

        # ======================================================
        # Evaluate
        # ======================================================
        results = self.evaluate_all_models(results)
        self.plot_results(results)
        self.plot_feature_importance(results)
        summary = self.save_summary(method, results)

        return summary

    # ======================================================
    # Evaluate One Model
    # ======================================================
    def evaluate_one_model(self, result):
        y_true = result["GroundTruth"]
        y_pred = result["Prediction"]
        y_score = result["Probability"]

        folder = result["Folder"]

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        report = classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )

        print()
        print("=" * 60)
        print(
            f"EVALUATION : "
            f"{result['Model'].upper()}"
        )
        print("=" * 60)

        print(
            f"Accuracy  : "
            f"{result['Accuracy']:.4f} "
            f"± {result['Accuracy STD']:.4f}"
        )

        print(
            f"Precision : "
            f"{result['Precision']:.4f} "
            f"± {result['Precision STD']:.4f}"
        )

        print(
            f"Recall    : "
            f"{result['Recall']:.4f} "
            f"± {result['Recall STD']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{result['F1']:.4f} "
            f"± {result['F1 STD']:.4f}"
        )

        print(
            f"ROC AUC   : "
            f"{result['AUC']:.4f} "
            f"± {result['AUC STD']:.4f}"
        )

        print()
        print(report)

        with open(
            folder / "classification_report.txt",
            "w"
        ) as f:

            f.write(report)

            f.write("\n\n")
            f.write(
                f"Accuracy Mean ± STD: "
                f"{result['Accuracy']:.6f} ± "
                f"{result['Accuracy STD']:.6f}\n"
            )

            f.write(
                f"Precision Mean ± STD: "
                f"{result['Precision']:.6f} ± "
                f"{result['Precision STD']:.6f}\n"
            )

            f.write(
                f"Recall Mean ± STD: "
                f"{result['Recall']:.6f} ± "
                f"{result['Recall STD']:.6f}\n"
            )

            f.write(
                f"F1 Mean ± STD: "
                f"{result['F1']:.6f} ± "
                f"{result['F1 STD']:.6f}\n"
            )

            f.write(
                f"ROC AUC Mean ± STD: "
                f"{result['AUC']:.6f} ± "
                f"{result['AUC STD']:.6f}\n"
            )

        result["ConfusionMatrix"] = cm

        return result

    # ======================================================
    # Evaluate All Models
    # ======================================================
    def evaluate_all_models(self, results):
        evaluation = []
        for result in results:
            result = self.evaluate_one_model(result)
            evaluation.append(result)
        return evaluation

    # ======================================================
    # Plot Confusion Matrix
    # ======================================================
    def plot_confusion_matrix(self, result):
        folder = result["Folder"]
        cm = result["ConfusionMatrix"]
        fig, ax = plt.subplots(figsize=(6, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Static",
                "Motion"
            ]
        )
        disp.plot(
            cmap="Blues",
            ax=ax,
            colorbar=False
        )
        plt.title(
            f"{result['Model']} Confusion Matrix"
        )
        plt.tight_layout()
        plt.savefig(
            folder/"confusion_matrix.png",
            dpi=300
        )
        plt.close()

    # ======================================================
    # Plot ROC
    # ======================================================
    def plot_roc_curve(self, result):
        if result["Probability"] is None:
            return
        y_true = result["GroundTruth"]
        y_score = result["Probability"]
        folder = result["Folder"]
        auc = result["AUC"]
        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )
        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {auc:.3f}"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            "--"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            f"{result['Model']} ROC"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            folder/"roc_curve.png",
            dpi=300
        )
        plt.close()

    # ======================================================
    # Plot Results
    # ======================================================
    def plot_results(self, results):
        for result in results:
            self.plot_confusion_matrix(
                result
            )
            self.plot_roc_curve(
                result
            )

    # ======================================================
    # Save Summary
    # ======================================================
    def save_summary(self, method, results):
        rows = []

        for result in results:
            rows.append({
                "Method": method,
                "Model": result["Model"],
                "Runtime(s)": result["Runtime"],

                "Accuracy": result["Accuracy"],
                "Accuracy STD": result["Accuracy STD"],

                "Precision": result["Precision"],
                "Precision STD": result["Precision STD"],

                "Recall": result["Recall"],
                "Recall STD": result["Recall STD"],

                "F1 Score": result["F1"],
                "F1 STD": result["F1 STD"],

                "ROC AUC": result["AUC"],
                "ROC AUC STD": result["AUC STD"],

                "Overall Score":
                    result["Overall Score"]
            })

        summary = pd.DataFrame(rows)

        summary = summary.sort_values(
            "Overall Score",
            ascending=False
        ).reset_index(drop=True)

        folder = CLASSIFICATION_DIR / method

        summary.to_csv(
            folder / "summary.csv",
            index=False
        )

        print()
        print("=" * 60)
        print(
            f"{method.upper()} SUMMARY"
        )
        print("=" * 60)
        print(summary)
        print("=" * 60)

        return summary

    # ======================================================
    # Classification Benchmark
    # ======================================================
    def classification_benchmark(self, summaries):
        benchmark = pd.concat(
            summaries,
            ignore_index=True
        )
        benchmark = benchmark.sort_values(
            "Overall Score",
            ascending=False
        )
        if "Rank" in benchmark.columns:
            benchmark.drop(columns=["Rank"], inplace=True)

        benchmark.insert(
            0,
            "Rank",
            np.arange(1, len(benchmark)+1)
        )
        benchmark.to_csv(
            BENCHMARK_DIR / "classification_benchmark.csv", index=False)
        print()
        print("="*60)
        print("CLASSIFICATION BENCHMARK")
        print("="*60)
        print(benchmark)
        print("="*60)

        return benchmark

    # ======================================================
    # Feature Importance
    # ======================================================
    def feature_importance(self, result):
        if result["Model"] != "random_forest":
            return

        folder = result["Folder"]

        importance_file = (
            folder / "feature_importance.csv"
        )

        if not importance_file.exists():
            print(
                f"Feature importance file not found: "
                f"{importance_file}"
            )
            return

        importance = pd.read_csv(
            importance_file
        )

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        # ======================================================
        # Save top 20
        # ======================================================
        top20 = importance.head(20)

        plt.figure(figsize=(8, 8))

        plt.barh(
            top20["Feature"][::-1],
            top20["Importance"][::-1]
        )

        plt.xlabel("Mean Feature Importance")
        plt.ylabel("Feature")

        plt.title(
            "Top 20 Feature Importance - Random Forest"
        )

        plt.tight_layout()

        plt.savefig(
            folder / "feature_importance.png",
            dpi=300
        )

        plt.close()

    # ======================================================
    # Plot Feature Importance
    # ======================================================
    def plot_feature_importance(self, results):
        for result in results:
            self.feature_importance(
                result
            )

    # ======================================================
    # Run
    # ======================================================
    def run(self):
        summaries = []
        for method in METHODS:
            summary = self.process(method)
            summaries.append(summary)
        self.classification_benchmark(summaries)


if __name__ == "__main__":
    print()
    print("="*60)
    print("MODULE 07 : CLASSIFICATION")
    print("="*60)
    pipeline = ClassificationPipeline()
    pipeline.run()
