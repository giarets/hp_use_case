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


def predict_last_13_weeks(df, fc_model, col_agg="sku"):
    df_last_13_weeks = select_last_n_weeks_from_df(df, n_weeks=13)
    X_test, y_test = df_last_13_weeks.drop(columns=["y"]), df_last_13_weeks["y"]
    y_preds = fc_model.predict(X=X_test)
    df_preds = pd.DataFrame(
        data={
            col_agg: X_test[col_agg],
            "y_pred": y_preds,
            # "y": y_test.values
        },
        index=X_test.index,
    )
    return df_preds


def aggregate_predictions(df, cols=['y_pred']):

    df_agg = df[["date", "id", "year_week", "product_number", "y"] + cols]
    df_agg = df_agg.copy()
    df_agg.loc[:, "date_temp"] = df_agg["date"]

    df_agg = df_agg.set_index("date")

    agg_dict = {
        "id": "first",
        "date_temp": "first",
        "year_week": "first",
        "product_number": "first",
        "y": "sum",
    }

    agg_dict.update({col: "sum" for col in cols})

    df_agg = (
        df_agg.groupby(["product_number"], observed=False)
        .resample("W")
        .agg(agg_dict)
        .reset_index(drop=True)
        .rename(columns={"date_temp": "date"})
    )
    return df_agg


def aggregate_df(df):

    df["date"] = pd.to_datetime(df["date"])

    df_grouped = df.groupby(
        ["date", "product_number", "reporterhq_id"], as_index=False
    ).agg(
        {
            "id": "first",
            "year_week": "first",
            "prod_category": "first",
            "specs": "first",
            "display_size": "first",
            "segment": "first",
            "inventory_units": "sum",
            "sales_units": "sum",
        }
    )

    # Pivot table to create separate columns for each reporter_id
    df_pivot = df_grouped.pivot_table(
        index=["date", "product_number"],
        columns="reporterhq_id",
        values=["inventory_units", "sales_units"],
        aggfunc="sum",
        fill_value=0,
    )

    df_pivot.columns = [f"{metric}_{reporter}" for metric, reporter in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    df_totals = df.groupby(["date", "product_number"], as_index=False)[
        ["inventory_units", "sales_units"]
    ].sum()
    df_final = df_totals.merge(df_pivot, on=["date", "product_number"])

    df_final = df_final.merge(
        df[[
            'id', 'date', 'year_week', 'product_number', 
            'reporterhq_id', 'prod_category', 'specs', 
            'display_size', 'segment'
            ]],
        on=['date', 'product_number'],
        how='left'
    )
    return df_final
