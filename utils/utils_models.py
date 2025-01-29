from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd
import utils.utils as utils
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


class AbstractForecastingModel(ABC):
    """An abstract base class for time-series forecasting models."""

    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.model = None
        self.model = self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, y_pred, y_test):
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        # print(f"Model evaluation completed. RMSE: {str(rmse)}")
        return rmse

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        model_and_metadata = {
            "model": self.model,
            "features": self.model.feature_name_,
            "hyperparameters": self.hyperparameters,
        }
        with open(filename, "wb") as file:
            pickle.dump(model_and_metadata, file)

    def load_model(self, filename):
        with open(filename, "rb") as file:
            loaded_data = pickle.load(file)
            self.model = loaded_data["model"]

    def cross_validate(self, df, n_splits=4, aggregate_by_id=False):
        metrics = []
        predictions_list = []

        unique_dates = pd.Series(df.index.unique())
        tss = TimeSeriesSplit(n_splits, test_size=13)

        for train_idx, test_idx in tss.split(unique_dates):
            train_dates, test_dates = (
                unique_dates.iloc[train_idx],
                unique_dates.iloc[test_idx],
            )

            train_data = df[df.index.isin(train_dates)]
            test_data = df[df.index.isin(test_dates)]

            X_train, y_train = train_data.drop(columns=["y"]), train_data["y"]
            X_test, y_test = test_data.drop(columns=["y"]), test_data["y"]

            self.train(X_train, y_train)
            y_pred = self.predict(X_test)

            score = self.evaluate(y_pred, y_test)
            metrics.append(score)

            predictions_df = pd.DataFrame({
                "date": test_data.index,
                "id": test_data['id'],
                "year_week": test_data['year_week'],
                "product_number": test_data['product_number'],
                "y_pred": y_pred,
                "y": y_test
            })
            predictions_list.append(predictions_df)

        average_rmse = np.mean(metrics)
        print(f"Average RMSE from cross-validation: {average_rmse:.4f}")

        if aggregate_by_id:
            df_final_preds = pd.concat(predictions_list).reset_index(drop=True)
            df_final_preds = utils.aggregate_predictions(df_final_preds)
            score_agg = self.evaluate(df_final_preds['y_pred'], df_final_preds['y'])
            print(f"\nAverage RMSE after aggregating per id: {score_agg:.4f}")

        return average_rmse
    

class NaiveRollingMean(AbstractForecastingModel):

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.window = None
        self.column = self.initialize_model()

    def initialize_model(self):
        if self.hyperparameters is None or 'window' not in self.hyperparameters:
            raise ValueError("Hyperparameter 'window' is required but missing.")
        self.window = self.hyperparameters['window']
        return f"inventory_units_rolling_mean_{self.window}w"

    def train(self, X_train, y_train):
        pass

    def predict(self, X):
        if self.column not in X.columns:
            raise ValueError(f"{self.column} is missing from input data.")

        last_values_per_sku = (
            X.sort_index()
            .groupby("sku")[self.column]
            .last()
            .to_dict()
        )
        return X["sku"].map(last_values_per_sku).values
    

class NaiveLag(AbstractForecastingModel):

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.lag = None
        self.column = self.initialize_model()

    def initialize_model(self):
        if self.hyperparameters is None or 'lag' not in self.hyperparameters:
            raise ValueError("Hyperparameter 'lag' is required but missing.")
        self.lag = self.hyperparameters['lag']
        return f"inventory_units_lag_{self.lag}"

    def train(self, X_train, y_train):
        pass

    def predict(self, X):
        if self.column not in X.columns:
            raise ValueError(f"{self.column} is missing from input data.")

        last_values_per_sku = (
            X.sort_index()
            .groupby("sku")[self.column]
            .last()
            .to_dict()
        )
        return X["sku"].map(last_values_per_sku).values
    

class LightGBMForecastingModel(AbstractForecastingModel):
    def initialize_model(self):
        return LGBMRegressor(**self.hyperparameters)
