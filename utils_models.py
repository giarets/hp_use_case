import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error


class ForecastingModel:

    def __init__(self, model_class=LGBMRegressor, hyperparameters=None):
        self.model_class = model_class
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.model = self.model_class(**self.hyperparameters)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, y_pred, y_test):
        rmse = round(root_mean_squared_error(y_test, y_pred), 3)
        print(f"Model evaluation completed. RMSE: {str(rmse)}")
        return rmse

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        model_and_metadata = {
            'model': self.model,
            'features': self.model.feature_name_,
            'hyperparameters': self.hyperparameters
        }
        with open(filename, 'wb') as file: 
            pickle.dump(model_and_metadata, file) 

    def load_model(self, filename):
        with open(filename, 'rb') as file: 
            self.model = pickle.load(file) 
    
    def cross_validate(self, df, unique_dates, n_splits=4):
        
        metrics = []
        tss = TimeSeriesSplit(n_splits, test_size=13)

        for train_idx, test_idx in tss.split(unique_dates):

            train_dates, test_dates = unique_dates.iloc[train_idx], unique_dates.iloc[test_idx]

            train_data = df[df.index.isin(train_dates)]
            test_data = df[df.index.isin(test_dates)]

            X_train, y_train = train_data.drop(columns=['y']), train_data['y']
            X_test, y_test = test_data.drop(columns=['y']), test_data['y']

            self.train(X_train, y_train)
            y_pred = self.predict(X_test)

            score = self.evaluate(y_pred, y_test)
            metrics.append(score)

        average_rmse = np.mean(metrics)
        print(f'Average RMSE from cross-validation: {average_rmse:.4f}')
        return average_rmse
