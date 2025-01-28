import pandas as pd
import numpy as np


def features_time_related(df):
    df['day_of_month'] = df['date'].dt.day
    df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1  # Week within the month
    df['week_of_year'] = df['date'].dt.isocalendar().week  # ISO week number
    df['month_of_year'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df['sin_day_of_month'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['cos_day_of_month'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    # Week of the year (1-52)
    df['sin_week_of_year'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week_of_year'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Month of the year (1-12)
    df['sin_month_of_year'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
    df['cos_month_of_year'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
    return df


def features_lag(df, col, lags=[13]):

    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby("sku", observed=False)[col].shift(lag)

    return df


def features_rolling(df, col, window_sizes):
    for window in window_sizes:
        df[f'{col}_rolling_mean_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).mean())
        df[f'{col}_rolling_std_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).std())
        df[f'{col}_rolling_sum_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).sum())
        # df[f'{col}_rolling_sum_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).min())
        # df[f'{col}_rolling_sum_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).max())
    return df


# copied
def create_periods_feature(df, group_columns, date_column, target_col):
        """
        Create a new feature 'feature_periods' that counts the number of weeks since
        the first non-zero signal for each group, based on the row order.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by (e.g., client, warehouse, product)
        - date_column: the column containing dates (e.g., 'date')
        - target_col: the column used to start counting when its value is greater than 0 (e.g., 'sales')

        Returns:
        - pandas DataFrame with new columns:
            - 'feature_periods' counting periods since the first non-zero signal,
            - 'feature_periods_expanding': expanded version of the periods count,
            - 'feature_periods_sqrt': square root of the periods count.
        """
        # Copy the input DataFrame
        df_copy = df.copy()

        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Ensure the date_column is in datetime format
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])

        # Sort by group_columns and the date_column to ensure proper order
        df_copy = df_copy.sort_values(by=group_columns + [date_column])

        # Create a mask to indicate rows where the signal_col is greater than 0
        df_copy['signal_above_zero'] = df_copy[target_col] > 0

        # Group by the group_columns and create a cumulative sum of the signal_above_zero mask
        # Start counting periods only when the signal_col is greater than 0
        df_copy['first_nonzero_signal'] = df_copy.groupby(group_columns)['signal_above_zero'].cumsum() > 0

        # Count periods only where the signal has been greater than zero
        df_copy['feature_periods'] = df_copy.groupby(group_columns).cumcount() + 1
        df_copy['feature_periods'] = df_copy['feature_periods'].where(df_copy['first_nonzero_signal'], 0)

        # Convert 'feature_periods' to float64
        df_copy['feature_periods'] = df_copy['feature_periods'].astype('float64')

        # Add feature_periods_expanding and feature_periods_sqrt
        df_copy['feature_periods_expanding'] = df_copy['feature_periods'] ** 1.10
        df_copy['feature_periods_sqrt'] = np.sqrt(df_copy['feature_periods'])

        # Ensure all new columns are float64
        df_copy['feature_periods_expanding'] = df_copy['feature_periods_expanding'].astype('float64')
        df_copy['feature_periods_sqrt'] = df_copy['feature_periods_sqrt'].astype('float64')

        # Reset the index after sorting and adding the feature_periods column
        df_copy = df_copy.reset_index(drop=True)

        # Drop the temporary columns
        df_copy = df_copy.drop(columns=['signal_above_zero', 'first_nonzero_signal'])

        return df_copy