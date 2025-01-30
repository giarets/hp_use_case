import pandas as pd

file_path = 'train_v_2_kaggle_23.csv'

date_present = pd.to_datetime('2023-02-04')

cols_inventory_lagged = [
    "inventory_units_lag_1",
    "inventory_units_lag_2",
    "inventory_units_lag_3",
    "inventory_units_lag_4",
    "inventory_units_lag_5",
    "inventory_units_lag_6",
    "inventory_units_lag_7",
    "inventory_units_lag_8",
    "inventory_units_lag_9",
    "inventory_units_lag_10",
    "inventory_units_lag_11",
    "inventory_units_lag_12",
]