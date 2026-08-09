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

    def blocked_split(self, dataset):
        test_indices = []
        train_indices = []

        purge = max(
            1,
            int(
                np.ceil(
                    WINDOW_OVERLAP / (1 - WINDOW_OVERLAP)
                )
            )
        )

        print()
        print("=" * 60)
        print("BLOCKED TEMPORAL SPLIT")
        print("=" * 60)
        print(
            f"Test size : {TEST_SIZE * 100:.1f}%"
        )
        print(
            f"Purge windows : {purge}"
        )
        print()

        # ======================================================
        # Split separately for STATIC and MOTION
        # ======================================================
        for label in sorted(dataset["Label"].unique()):

            class_indices = np.where(
                dataset["Label"].values == label
            )[0]

            n = len(class_indices)

            if n < 5:
                raise ValueError(
                    f"Not enough samples for label {label}: {n}"
                )

            # --------------------------------------------------
            # Test block = last TEST_SIZE portion
            # --------------------------------------------------
            n_test = max(
                1,
                int(np.ceil(n * TEST_SIZE))
            )

            test_start = n - n_test

            # --------------------------------------------------
            # Purge samples immediately before test
            # --------------------------------------------------
            train_end = max(
                0,
                test_start - purge
            )

            train_class = class_indices[:train_end]
            test_class = class_indices[test_start:]

            train_indices.extend(
                train_class.tolist()
            )

            test_indices.extend(
                test_class.tolist()
            )

            print(
                f"Label {label}: "
                f"Train={len(train_class)}, "
                f"Test={len(test_class)}, "
                f"Purged={test_start - train_end}"
            )

        train_indices = np.array(
            train_indices,
            dtype=int
        )

        test_indices = np.array(
            test_indices,
            dtype=int
        )

        return train_indices, test_indices

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
            model = SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            )
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
    
            train_idx = data["train_idx"]
            test_idx = data["test_idx"]
    
            dataset = data["dataset"]
            out_dir = data["out_dir"]
    
            # ======================================================
            # Create output folder
            # ======================================================
            folder = (out_dir/ model_name)
            folder.mkdir(parents=True,exist_ok=True)
    
            # ======================================================
            # Raw train/test
            # ======================================================
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
    
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
    
            # ======================================================
            # SCALER
            # ======================================================
            scaler = StandardScaler()
    
            # Fit ONLY on training data
            X_train = scaler.fit_transform(X_train)
    
            # Transform test using training scaler
            X_test = scaler.transform(X_test)
    
            # ======================================================
            # Model
            # ======================================================
            model = self.create_model(model_name)
            start = time.perf_counter()
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            runtime = (time.perf_counter()- start)
    
            # ======================================================
            # Probability
            # ======================================================
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = None
    
            # ======================================================
            # Save predictions
            # ======================================================
            prediction_df = pd.DataFrame({
                "Sample": dataset.iloc[test_idx]["Sample"].values,
    
                "GroundTruth": y_test.values,
    
                "Prediction": y_pred
            })
    
            prediction_df.to_csv(
                folder / "predictions.csv",
                index=False
            )
    
            # ======================================================
            # Result
            # ======================================================
            return {
                "Model": model_name,
                "Folder": folder,
                "GroundTruth": y_test.values,
                "Prediction": y_pred,
                "Probability": y_score,
                "Runtime(s)": runtime,
                "ModelObject": model,
                "FeatureNames": data["feature_names"]
            }

    # ======================================================
    # Train All Models
    # ======================================================
    def train_all_models(self, data):
        results = []
        models = self.get_models()
        for model_name, model in models.items():
            result = self.train_one_model(model_name, data)
            results.append(result)
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
        out_dir = self.save_dataset(method,dataset)
        self.dataset_information(dataset)

        # ======================================================
        # Prepare
        # ======================================================
        X, y, feature_names = self.prepare_data(dataset)

        # ======================================================
        # Leakage-safe split
        # ======================================================
        train_idx, test_idx = self.blocked_split(dataset)

        # ======================================================
        # Data
        # ======================================================
        data = {
            "dataset": dataset,
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "train_idx": train_idx,
            "test_idx": test_idx,
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
        print()
        print("=" * 60)
        print(f"EVALUATION : {result['Model'].upper()}")
        print("=" * 60)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
        if y_score is not None:
            auc = roc_auc_score(
                y_true,
                y_score
            )
        else:
            auc = np.nan
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

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print()
        print(report)
        with open(folder / "classification_report.txt", "w") as f:
            f.write(report)
        result["Accuracy"] = accuracy
        result["Precision"] = precision
        result["Recall"] = recall
        result["F1"] = f1
        result["AUC"] = auc
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
            overall = (result["Accuracy"] + result["Precision"] +
                       result["Recall"] + result["F1"]) / 4
            rows.append({
                "Method": method,
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1"],
                "ROC AUC": result["AUC"],
                "Overall Score": overall
            })

        summary = pd.DataFrame(rows)
        folder = CLASSIFICATION_DIR / method
        summary = summary.sort_values(
            "Overall Score",
            ascending=False
        ).reset_index(drop=True)

        summary.to_csv(
            folder / "summary.csv",
            index=False
        )
        print()
        print("="*60)
        print(summary)
        print("="*60)
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
    def feature_importance(
        self,
        result
    ):
        if result["Model"] != "random_forest":
            return
        model = result["ModelObject"]
        folder = result["Folder"]
        feature_names = result["FeatureNames"]
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })
        importance = importance.sort_values(
            "Importance",
            ascending=False
        )
        importance.to_csv(
            folder / "feature_importance.csv",
            index=False
        )
        plt.figure(figsize=(8, 8))
        plt.barh(
            importance["Feature"][:20],
            importance["Importance"][:20]
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importance")
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
