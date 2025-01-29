import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_real_vs_predicted(
    df, single_ts, pred_columns, col_agg="sku", vline_dates=None
):
    """
    Plot real vs multiple predicted inventory values for a specific SKU.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the data with at least the following columns:
                            - 'sku': The SKU identifier.
                            - 'date': The date of the record.
                            - Predicted columns: A list of column names containing predicted values (e.g., 'y_pred_lag', 'y_pred_rolling').
                            - 'inventory_units' or 'y': Actual inventory values or alternative actual column.
    sku (str): The SKU identifier for which the comparison should be plotted.
    pred_columns (list of str): List of column names containing predicted values to plot.
    full_history (bool, optional): If set to `True`, the function plots the whole time series of
                                   'inventory_units' along with predicted values.
    vline_dates (list, optional): A list of dates (as strings or datetime objects) where vertical lines should be added.

    Returns:
    None: The function directly displays a plot. It does not return a value.
    """

    df_single_sku = df[df[col_agg] == single_ts].set_index("date")
    col_real = "y"

    plt.figure(figsize=(12, 4))

    # Loop over each prediction column and plot it
    for pred_col in pred_columns:
        plt.plot(
            df_single_sku.index,
            df_single_sku[pred_col],
            label=f"Predicted ({pred_col})",
            linestyle="--",
            marker="o",
            linewidth=1,
        )

    # Plot actual values
    plt.plot(
        df_single_sku.index,
        df_single_sku[col_real],
        label="Inventory Units",
        linestyle="-.",
        marker="^",
        color="green",
        linewidth=1,
    )

    # Add vertical lines if any
    if vline_dates:
        for vline_date in vline_dates:
            plt.axvline(
                x=vline_date,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Last available date" if vline_date == vline_dates[0] else "",
            )

    plt.title(f"Actual vs Predicted for {col_agg} {single_ts}")
    plt.xlabel("Date")
    plt.ylabel("Inventory Units")

    plt.legend()
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()


def plot_time_series_split_with_dates(ts_splitter, df):
    """
    Visualize the training and testing indices for a time series dataset.

    Parameters:
    - ts_splitter: Cross-validator (e.g., TimeSeriesSplit)
    - df: Dataset with a datetime index
    """
    fig, ax = plt.subplots(figsize=(12, ts_splitter.n_splits + 2))
    colors = {"training": "blue", "testing": "orange"}

    df = pd.Series(
        range(len(df.index.unique())), index=df.index.unique(), name="original_dates"
    )

    df = df[~df.index.duplicated(keep="first")]
    dates = df.index

    for split_idx, (train_idx, test_idx) in enumerate(ts_splitter.split(df)):
        # Scatter training indices
        ax.scatter(
            dates[train_idx],
            [split_idx] * len(train_idx),
            c=colors["training"],
            label="Training" if split_idx == 0 else "",
        )
        # Scatter testing indices
        ax.scatter(
            dates[test_idx],
            [split_idx] * len(test_idx),
            c=colors["testing"],
            label="Testing" if split_idx == 0 else "",
        )

    ax.legend(loc="upper left")
    ax.set_title("Time Series Split Visualization")
    ax.set_xlabel("Date")
    ax.set_ylabel("")
    ax.set_yticks(range(ts_splitter.n_splits))
    ax.set_yticklabels([f"Split {i+1}" for i in range(ts_splitter.n_splits)])
    plt.xticks(rotation=45)
    plt.figure(figsize=(14, 2))
    plt.tight_layout()
    plt.show()


def categorical_histogram(df, cat_col, log_scale=False):
    """
    Seaborn multiple histograms for a given categorical column.
    """

    col_sales = "sales_units"
    col_inventory = "inventory_units"

    if log_scale:
        df.loc[:, "sales_units_log"] = np.log1p(df["sales_units"])
        df.loc[:, "inventory_units_log"] = np.log1p(df["inventory_units"])
        col_sales = "sales_units_log"
        col_inventory = "inventory_units_log"

    g = sns.FacetGrid(df, col=cat_col, col_wrap=4, sharex=False, sharey=False, height=4)

    for ax, (name, data) in zip(g.axes.flat, df.groupby(cat_col)):
        sns.histplot(
            data[col_sales],
            ax=ax,
            kde=False,
            color="blue",
            alpha=0.3,
            label="sales_units",
            bins=25,
        )
        sns.histplot(
            data[col_inventory],
            ax=ax,
            kde=False,
            color="orange",
            alpha=0.3,
            label="inventory_units",
            bins=40,
        )
        ax.legend()

    g.set_titles("Segment: {col_name}")
    plt.tight_layout()
    plt.show()
