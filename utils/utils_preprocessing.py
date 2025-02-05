import pandas as pd
import utils.utils_preprocessing as utils_preprocessing
import utils.utils_features as utils_features


def load_and_preprocess(file_path, bottom_up=True, group_col="sku"):
    """
    Load a CSV file, preprocess its columns, fill in missing dates, interpolate
    missing values, generate model features and return the processed DataFrame.
    """
    df_kaggle = (
        pd.read_csv(file_path)
        .pipe(utils_preprocessing.preprocess_columns, bottom_up=bottom_up)
        .pipe(utils_preprocessing.fill_in_missing_dates, group_col=group_col)
        .pipe(
            lambda df: df.groupby("sku", group_keys=False, observed=False).apply(
                utils_preprocessing.interpolate
            )
        )
        .pipe(utils_features.features_time_related)
        .pipe(utils_features.features_lag, col="inventory_units", lags=[13, 14, 15])
        .pipe(utils_features.features_lag, col="sales_units", lags=[13, 14, 15])
        .pipe(
            utils_features.features_rolling, col="inventory_units", window_sizes=[4, 8]
        )
        .pipe(utils_features.features_rolling, col="sales_units", window_sizes=[4, 8])
        .pipe(
            utils_features.create_periods_feature,
            coll_agg="sku",
            date_column="date",
            target_col="inventory_units",
        )
        .rename(columns={"inventory_units": "y"})
        .set_index("date")
        .sort_index()
        .dropna()
    )

    return df_kaggle


def preprocess_columns(df, bottom_up=True):
    """
    Different preprocessing depending if the approach is
    bottom-up or not.
    """

    if bottom_up:
        # Define sku = reporterhq_id + product_number
        df["sku"] = (
            df["reporterhq_id"].astype(str) + "_" + df["product_number"].astype(str)
        )

    # Format columns
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    numeric_cols = [x for x in df.columns if "units" in x]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert to categorical
    categorical_columns = [
        "id",
        "product_number",
        "prod_category",
        "display_size",
        "segment",
    ]
    if bottom_up:
        categorical_columns += ["sku", "reporterhq_id"]

    for col in categorical_columns:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.remove_unused_categories()

    # Drop columns
    df.drop(columns=["specs"], inplace=True)
    df = df.dropna(subset=["inventory_units"])
    return df


def preprocess_columns_product_level(df):

    # Format columns
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["sales_units"] = pd.to_numeric(df["sales_units"], errors="coerce")
    df["inventory_units"] = pd.to_numeric(df["inventory_units"], errors="coerce")

    # Convert to categorical
    categorical_columns = [
        "id",
        "product_number",
        "reporterhq_id",
        "prod_category",
        "display_size",
        "segment",
    ]
    for col in categorical_columns:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.remove_unused_categories()

    # Drop columns
    df.drop(columns=["specs"], inplace=True)
    df = df.dropna(subset=["inventory_units"])

    #
    [x for x in df.columns if "units" in x]


def fill_in_missing_dates(df, group_col="sku", date_col="date", freq="W-SAT"):
    """
    Ensure that each group has all dates with a specified frequency from its 
    min to its max date. Missing rows will be forward-filled except for sales_units 
    and inventory_units which will have NaN values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    group_col (list): Columns to group by.
    date_col (str): The name of the date column.
    freq (str): Frequency string for the date range (e.g., 'W-SAT' for weekly on Saturdays).

    Returns:
    pd.DataFrame: The completed DataFrame with all dates for each group.
    """
    if date_col not in df.columns:
        df = df.reset_index()

    df[date_col] = pd.to_datetime(df[date_col])
    original_dtype = df[group_col].dtype
    df_dates_ranges = (
        df.groupby(group_col, observed=False)[date_col]
        .agg(["min", "max"])
        .reset_index()
    )

    df_complete = pd.DataFrame()

    # Generate all required dates for each group based on the specified frequency
    for _, row in df_dates_ranges.iterrows():
        dates = pd.date_range(start=row["min"], end=row["max"], freq=freq)

        # Create a DataFrame for this group with all dates
        temp_df = pd.DataFrame({**row.drop(["min", "max"]).to_dict(), date_col: dates})
        df_complete = pd.concat([df_complete, temp_df], ignore_index=True)

    df_complete = pd.merge(df_complete, df, on=[group_col] + [date_col], how="left")

    exclude_columns = ["sales_units", "inventory_units"]
    fill_columns = [
        col
        for col in df_complete.columns
        if col not in exclude_columns + [group_col, date_col]
    ]
    df_complete[fill_columns] = df_complete.groupby(group_col)[fill_columns].ffill()

    if pd.api.types.is_categorical_dtype(original_dtype):
        df_complete[group_col] = df_complete[group_col].astype("category")

    return df_complete


def interpolate(group):
    
    group = group.sort_values(by="date")
    group = group.set_index("date")
    group["sales_units"] = group["sales_units"].interpolate(method="time")
    group["inventory_units"] = group["inventory_units"].interpolate(method="time")
    return group.reset_index()
