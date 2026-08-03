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
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
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
# CONFIG
# ==========================================================
METHODS = [
    "butterworth",
    "median",
    "moving_average"
]

MODELS = [
    "logistic",
    "svm",
    "random_forest",
    "mlp"
]
RANDOM_STATE = 42
TEST_SIZE = 0.2


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
    def prepare_data(self,dataset):
        y = dataset["Label"]
        X = dataset.drop(
            columns=[
                "Sample",
                "Label",
                "Method",
                "Dataset"
            ],
            errors="ignore"
        )

        X = X.astype(np.float64)
        feature_names = X.columns.tolist()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return (X, y, feature_names, scaler)

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
    # Cross Validation
    # ======================================================
    def create_cv(self, X, y):
        n_sample = len(y)
        print()
        print("="*60)
        print(f"Total Samples : {n_sample}")
        print("="*60)

        # --------------------------------------------
        # Very Small Dataset
        # --------------------------------------------
        if n_sample < 10:
            print("Validation : Leave-One-Out")
            cv = LeaveOneOut()

        # --------------------------------------------
        # Small Dataset
        # --------------------------------------------
        elif n_sample < 20:
            print("Validation : Stratified KFold (5)")
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE
            )

        # --------------------------------------------
        # Normal Dataset
        # --------------------------------------------
        else:
            print("Validation : Train/Test Split")
            cv = None
        return cv

    # ======================================================
    # Split Dataset
    # ======================================================
    def split_dataset(self, X, y):
        if len(y) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
            return (X_train, X_test, y_train, y_test)
        return None

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
    def train_one_model(self, model_name, model, data):
        self.print_model_information(
            model_name,
            model
        )

        X = data["X"]
        y = data["y"]
        feature_names = data["feature_names"]
        out_dir = data["out_dir"]
        folder = self.create_output_folder(
            out_dir,
            model_name
        )
        cv = data["cv"]
        split = data["split"]
        sample_name = data["dataset"]["Sample"].values
        print("Training...")
        start = time.time()

        # ==============================================
        # Train/Test
        # ==============================================
        if split is not None:
            X_train, X_test, y_train, y_test = split
            model.fit(
                X_train,
                y_train
            )
            y_pred = model.predict(
                X_test
            )
            if hasattr(
                model,
                "predict_proba"
            ):
                y_score = model.predict_proba(
                    X_test
                )[:, 1]
            else:
                y_score = None
            sample_test = sample_name[
                y_test.index
            ]

        # ==============================================
        # Cross Validation
        # ==============================================
        else:
            y_true = []
            y_pred = []
            y_score = []

            sample_test = []
            for train_idx, test_idx in cv.split(X, y):
                X_train = X[train_idx]
                X_test = X[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                model.fit(
                    X_train,
                    y_train
                )
                pred = model.predict(
                    X_test
                )
                y_true.extend(
                    y_test.tolist()
                )
                y_pred.extend(
                    pred.tolist()
                )
                sample_test.extend(
                    sample_name[test_idx]
                )
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_test)[:, 1]
                    y_score.extend(
                        prob.tolist()
                    )

            y_test = np.array(
                y_true
            )
            y_pred = np.array(
                y_pred
            )
            if len(y_score) > 0:
                y_score = np.array(
                    y_score
                )
            else:
                y_score = None

        runtime = time.time() - start

        print(f"Runtime : {runtime:.3f} s")

        self.save_prediction(folder, sample_test, y_test, y_pred, y_score)

        self.save_model(folder, model)

        result = {
            "Method": data["dataset"]["Method"].iloc[0],
            "Model": model_name,
            "Runtime": runtime,
            "ModelObject": model,
            "Folder": folder,
            "FeatureNames": feature_names,
            "Sample": sample_test,
            "GroundTruth": y_test,
            "Prediction": y_pred,
            "Probability": y_score
        }

        return result

    # ======================================================
    # Train All Models
    # ======================================================
    def train_all_models(self, data):
        results = []
        models = self.get_models()
        for model_name, model in models.items():
            result = self.train_one_model(
                model_name,
                model,
                data
            )
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

        # ------------------------------------------
        # Load dataset
        # ------------------------------------------
        dataset = self.load_dataset(method)
        out_dir = self.save_dataset(
            method,
            dataset
        )
        self.dataset_information(dataset)

        # ------------------------------------------
        # Prepare data
        # ------------------------------------------
        X, y, feature_names, scaler = self.prepare_data(dataset)
        cv = self.create_cv(
            X,
            y
        )
        split = self.split_dataset(
            X,
            y
        )

        # ------------------------------------------
        # Create data dictionary
        # ------------------------------------------
        data = {
            "dataset": dataset,
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "scaler": scaler,
            "cv": cv,
            "split": split,
            "out_dir": out_dir
        }

        # ------------------------------------------
        # Train all models
        # ------------------------------------------
        results = self.train_all_models(data)
        results = self.evaluate_all_models(results)
        self.plot_results(results)

        self.plot_feature_importance(results)

        summary = self.save_summary(
            method,
            results
        )

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
            result = self.evaluate_one_model(
                result
            )
            evaluation.append(
                result
            )
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
                "Runtime(s)": result["Runtime"],
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
            np.arange(1,len(benchmark)+1)
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
