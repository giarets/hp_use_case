import pandas as pd
import numpy as np


def looks_for_missing_dates(df, freq="7D"):

    for id_, group in df.set_index("date").groupby("sku"):
        expected_dates = pd.date_range(
            start=group.index.min(), end=group.index.max(), freq=freq
        )
        actual_dates = group.index
        missing_dates = expected_dates.difference(actual_dates)

        if not missing_dates.empty:
            print(f"Missing dates for ID {id_}: {missing_dates.tolist()}")


def train_test_split(df, forecasting_horizon=13, target_col="y"):

    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()
    split_date = df.index.max() - pd.DateOffset(weeks=forecasting_horizon - 1)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    return X_train, X_test, y_train, y_test


def select_last_n_weeks_from_df(df, n_weeks):
    time_index = df.index
    time_mask = time_index > time_index.max() - np.timedelta64(n_weeks, "W")
    return df.loc[time_mask]


def predict_last_13_weeks(df, fc_model):
    df_last_13_weeks = select_last_n_weeks_from_df(df, n_weeks=13)
    X_test, y_test = df_last_13_weeks.drop(columns=["y"]), df_last_13_weeks["y"]
    y_preds = fc_model.model.predict(X=X_test)
    df_preds = pd.DataFrame(
        data={
            "sku": X_test["sku"],
            "y_pred": y_preds,
            # "y": y_test.values
        },
        index=X_test.index,
    )
    return df_preds
