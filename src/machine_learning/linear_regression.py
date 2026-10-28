"""Linear regression examples for advertising spend and sales."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _fit_and_score(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[LinearRegression, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return model, metrics


def train_sales_model(
    dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[LinearRegression, dict[str, float]]:
    """Fit a multiple linear regression model and score its holdout split."""
    normalized = dataframe.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()
    required = {"tv", "radio", "newspaper", "sales"}
    missing_columns = sorted(required - set(normalized.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if normalized.empty:
        raise ValueError("CSV contains no rows.")
    return _fit_and_score(
        normalized[["tv", "radio", "newspaper"]],
        normalized["sales"],
        test_size,
        random_state,
    )


def train_tv_sales_model(
    dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[LinearRegression, dict[str, float]]:
    """Fit a single-feature model using TV advertising spend."""
    normalized = dataframe.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()
    if "tv" not in normalized or "sales" not in normalized:
        raise ValueError("CSV must contain 'TV' and 'sales' columns.")
    if normalized.empty:
        raise ValueError("CSV contains no rows.")
    return _fit_and_score(
        normalized[["tv"]],
        normalized["sales"],
        test_size,
        random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the advertising CSV file")
    parser.add_argument("--no-plots", action="store_true", help="Skip the TV/sales plot")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.data)
    normalized = dataframe.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()
    tv_model, tv_metrics = train_tv_sales_model(normalized)
    multiple_model, multiple_metrics = train_sales_model(normalized)
    for label, model, metrics in (
        ("TV-only model", tv_model, tv_metrics),
        ("Multiple regression model", multiple_model, multiple_metrics),
    ):
        print(f"\n{label} holdout metrics")
        for name, value in metrics.items():
            print(f"{name}: {value:.3f}")
        print("Intercept:", round(float(model.intercept_), 3))
        print("Coefficients:", dict(zip(model.feature_names_in_, model.coef_.round(3))))

    if not args.no_plots:
        sns.regplot(data=normalized, x="tv", y="sales", ci=None)
        plt.title("Sales vs TV advertising")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
