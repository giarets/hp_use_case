import pandas as pd
import matplotlib.pyplot as plt

def plot_real_vs_predicted(df, sku, full_history=False):
    """
    Plot the comparison between real (actual) and predicted inventory values for a specific SKU.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the data with at least the following columns:
                            - 'sku': The SKU identifier.
                            - 'date': The date of the record.
                            - 'y_pred': Predicted values (if available).
                            - 'inventory_units' or 'y': Actual inventory values or alternative actual column.
    sku (str): The SKU identifier for which the comparison should be plotted.
    full_history (bool, optional): If set to `True`, the function plots the whole time series of 
                                   'inventory_units' along with predicted values.

    Returns:
    None: The function directly displays a plot. It does not return a value.
    
    Example:
    plot_real_vs_predicted(df, sku="12_112518", full_history=True)
    """
    
    df_single_sku = df[df['sku'] == sku].set_index('date')

    col_predictions = 'y_pred'
    col_real = 'inventory_units' if full_history else 'y'

    plt.figure(figsize=(12, 4))

    # Plot predicted values ('y_pred')
    plt.plot(df_single_sku.index, df_single_sku[col_predictions], 
             label='Predicted (y_pred)', linestyle='--', marker='o', color='orange', linewidth=1)

    # Plot actual values ('y' or 'inventory_units')
    plt.plot(df_single_sku.index, df_single_sku[col_real], 
             label='Inventory Units', linestyle='-.', marker='^', color='green', linewidth=1)

    plt.title(f"Actual vs Predicted for sku {sku}")
    plt.xlabel("Date")
    plt.ylabel("Inventory Units")
    
    plt.legend()
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()
