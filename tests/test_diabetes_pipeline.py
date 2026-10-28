import pandas as pd
import pytest

from machine_learning.diabetes_pipeline import (
    load_diabetes_data,
    train_and_evaluate,
)


def make_diabetes_data(rows: int = 100) -> pd.DataFrame:
    values = {
        "Pregnancies": [index % 8 for index in range(rows)],
        "Glucose": [90 + index % 90 for index in range(rows)],
        "BloodPressure": [60 + index % 30 for index in range(rows)],
        "SkinThickness": [15 + index % 20 for index in range(rows)],
        "Insulin": [50 + index % 100 for index in range(rows)],
        "BMI": [20 + index % 15 for index in range(rows)],
        "DiabetesPedigreeFunction": [0.1 + (index % 10) / 10 for index in range(rows)],
        "Age": [20 + index % 50 for index in range(rows)],
        "Outcome": [index % 2 for index in range(rows)],
    }
    return pd.DataFrame(values)


def test_training_returns_metrics_and_expected_shapes() -> None:
    model, metrics, matrix, report = train_and_evaluate(make_diabetes_data())

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())
    assert matrix.shape == (2, 2)
    assert "precision" in report
    prediction_data = make_diabetes_data().drop(columns="Outcome")
    prediction_data.columns = prediction_data.columns.str.lower()
    assert len(model.predict(prediction_data)) == 100


def test_zero_values_are_handled_by_pipeline() -> None:
    dataframe = make_diabetes_data()
    dataframe.loc[0, ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = 0

    _, metrics, _, _ = train_and_evaluate(dataframe)

    assert all(pd.notna(value) for value in metrics.values())


def test_loader_reports_missing_columns(tmp_path) -> None:
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"Outcome": [0, 1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_diabetes_data(path)


def test_loader_normalizes_column_names(tmp_path) -> None:
    path = tmp_path / "diabetes.csv"
    make_diabetes_data().to_csv(path, index=False)

    dataframe = load_diabetes_data(path)

    assert "outcome" in dataframe.columns
    assert all(column == column.lower() for column in dataframe.columns)


def test_training_rejects_non_binary_outcomes() -> None:
    dataframe = make_diabetes_data()
    dataframe["Outcome"] = 2

    with pytest.raises(ValueError, match="binary labels 0 and 1"):
        train_and_evaluate(dataframe)
