import pandas as pd

def looks_for_missing_dates(df, freq='7D'):

    for id_, group in df.set_index('date').groupby('sku'):
        expected_dates = pd.date_range(start=group.index.min(), end=group.index.max(), freq=freq)
        actual_dates = group.index
        missing_dates = expected_dates.difference(actual_dates)

        if not missing_dates.empty:
            print(f"Missing dates for ID {id_}: {missing_dates.tolist()}")
        


def fill_in_missing_dates(df, group_col='sku', date_col='date', freq='W-SAT'):
    """
    Ensure that each group has all dates with a specified frequency from its min to its max date. 
    Missing rows will be forward-filled except for sales_units and inventory_units which will have NaN values.

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
    df_dates_ranges = df.groupby(group_col)[date_col].agg(['min', 'max']).reset_index()

    df_complete = pd.DataFrame()

    # Generate all required dates for each group based on the specified frequency
    for _, row in df_dates_ranges.iterrows():
        dates = pd.date_range(start=row['min'], end=row['max'], freq=freq)

        # Create a DataFrame for this group with all dates
        temp_df = pd.DataFrame({
            **row.drop(['min', 'max']).to_dict(),
            date_col: dates
        })
        df_complete = pd.concat([df_complete, temp_df], ignore_index=True)

    df_complete = pd.merge(df_complete, df, on=[group_col] + [date_col], how='left')

    exclude_columns = ['sales_units', 'inventory_units']
    fill_columns = [col for col in df_complete.columns if col not in exclude_columns + [group_col, date_col]]
    df_complete[fill_columns] = df_complete.groupby(group_col)[fill_columns].ffill()

    if pd.api.types.is_categorical_dtype(original_dtype):
        df_complete[group_col] = df_complete[group_col].astype('category')

    return df_complete


def train_test_split(df, forecasting_horizon=13, target_col='y'):
    
    if 'date' in df.columns:
        df = df.set_index('date')
    df = df.sort_index()
    split_date = df.index.max() - pd.DateOffset(weeks=forecasting_horizon -1)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    return X_train, X_test, y_train, y_test