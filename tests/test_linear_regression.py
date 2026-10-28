import pandas as pd

from machine_learning.linear_regression import (
    train_sales_model,
    train_tv_sales_model,
)


def make_advertising_data(rows: int = 50) -> pd.DataFrame:
    tv = [float(index * 2) for index in range(rows)]
    radio = [float(index % 10) for index in range(rows)]
    newspaper = [float(index % 7) for index in range(rows)]
    return pd.DataFrame(
        {
            "TV": tv,
            "radio": radio,
            "newspaper": newspaper,
            "sales": [2 + 0.05 * tv[i] + 0.3 * radio[i] for i in range(rows)],
        }
    )


def test_both_sales_models_train_and_report_holdout_metrics() -> None:
    dataframe = make_advertising_data()

    tv_model, tv_metrics = train_tv_sales_model(dataframe)
    multiple_model, multiple_metrics = train_sales_model(dataframe)

    assert list(tv_model.feature_names_in_) == ["tv"]
    assert list(multiple_model.feature_names_in_) == ["tv", "radio", "newspaper"]
    assert set(tv_metrics) == {"rmse", "mae", "r2"}
    assert set(multiple_metrics) == {"rmse", "mae", "r2"}
