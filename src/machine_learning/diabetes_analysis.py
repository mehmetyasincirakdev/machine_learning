"""Exploratory analysis helpers for the diabetes dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from machine_learning.diabetes_pipeline import load_diabetes_data


def analyze_diabetes(dataframe: pd.DataFrame, show_plots: bool = True) -> None:
    """Print dataset summaries and optionally display exploratory plots."""
    print(f"Rows: {len(dataframe)} | Columns: {len(dataframe.columns)}")
    print("\nColumn types")
    print(dataframe.dtypes)
    print("\nMissing values")
    print(dataframe.isna().sum())
    print("\nSummary statistics")
    print(dataframe.describe().T)
    print("\nOutcome counts")
    print(dataframe["outcome"].value_counts().sort_index())

    if show_plots:
        numeric_columns = dataframe.select_dtypes(include="number").columns
        dataframe[numeric_columns].hist(figsize=(12, 9), bins=20)
        plt.tight_layout()
        plt.show()

        sns.heatmap(dataframe[numeric_columns].corr(), annot=True, cmap="RdBu_r", center=0)
        plt.title("Feature correlation")
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the diabetes CSV file")
    parser.add_argument("--no-plots", action="store_true", help="Print summaries without plots")
    args = parser.parse_args()
    dataframe = load_diabetes_data(Path(args.data))
    analyze_diabetes(dataframe, show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
