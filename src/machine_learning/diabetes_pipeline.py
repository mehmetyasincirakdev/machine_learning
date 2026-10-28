"""Leakage-safe logistic regression workflow for the diabetes dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ZERO_AS_MISSING = ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]
OTHER_FEATURES = ["pregnancies", "diabetespedigreefunction", "age"]
REQUIRED_COLUMNS = [*OTHER_FEATURES, *ZERO_AS_MISSING, "outcome"]


def load_diabetes_data(path: str | Path) -> pd.DataFrame:
    """Load a diabetes CSV and normalize column names."""
    dataframe = pd.read_csv(path)
    dataframe.columns = dataframe.columns.str.strip().str.lower()
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if dataframe.empty:
        raise ValueError("CSV contains no rows.")
    return dataframe


def _replace_zeros(values: np.ndarray) -> np.ndarray:
    return np.where(values == 0, np.nan, values)


def build_diabetes_pipeline() -> Pipeline:
    """Build preprocessing and classification steps fitted only on training data."""
    zero_missing_pipeline = Pipeline(
        steps=[
            ("replace_zero", FunctionTransformer(_replace_zeros, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("zero_missing", zero_missing_pipeline, ZERO_AS_MISSING),
            ("numeric", numeric_pipeline, OTHER_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_and_evaluate(
    dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline, dict[str, float], np.ndarray, str]:
    """Fit the diabetes classifier and return metrics, confusion matrix, and report."""
    normalized = dataframe.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(normalized.columns))
    if missing_columns:
        raise ValueError(f"Dataframe is missing required columns: {', '.join(missing_columns)}")
    if normalized.empty:
        raise ValueError("Dataframe contains no rows.")

    features = normalized[OTHER_FEATURES + ZERO_AS_MISSING]
    target = normalized["outcome"]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("The outcome column must contain only the binary labels 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    model = build_diabetes_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return (
        model,
        metrics,
        confusion_matrix(y_test, predictions),
        classification_report(y_test, predictions, zero_division=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the diabetes CSV file")
    args = parser.parse_args()

    dataframe = load_diabetes_data(args.data)
    _, metrics, matrix, report = train_and_evaluate(dataframe)
    print("Test metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    print("\nConfusion matrix")
    print(matrix)
    print("\nClassification report")
    print(report)


if __name__ == "__main__":
    main()
