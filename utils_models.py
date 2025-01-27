import pickle
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error


class ForecastingModel:

    def __init__(self, model_class=LGBMRegressor, hyperparameters={}):
        self.model_class = model_class
        self.hyperparameters = hyperparameters
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

    @staticmethod
    def _train_test_split(df, forecasting_horizon=13, target_col='y'):
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
