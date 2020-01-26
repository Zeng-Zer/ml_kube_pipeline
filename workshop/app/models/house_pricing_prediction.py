import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
import os
import joblib
import json

# https://scikit-learn.org/stable/modules/classes.html#regression-metrics

class HousePricingPrediction:
    def __init__(self):
        self._clf = None

    def train(self):
        data_path = os.getenv('DATA_PATH')
        trained_model_path = os.getenv('TRAINED_MODEL_PATH')

        if data_path is None:
            raise RuntimeError('Data path must not be none')

        data = pd.read_csv(data_path)
        train_data = data.drop(['id', 'price'], axis=1)
        labels = data['price']
        x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.10, random_state=2)
        self._clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                                 learning_rate=0.1, loss='ls')
        self._clf.fit(x_train, y_train)

        y_predict = self._clf.predict(x_test)
        mse = mean_squared_error(y_test, y_predict)
        evs = explained_variance_score(y_test, y_predict)
        r2s = r2_score(y_test, y_predict)
        metrics = {
            'metrics': [
                {
                    'name': 'mean-squared-error',
                    'numberValue':  mse,
                    'format': "RAW"
                },
                {
                    'name': 'explained-variance-score',
                    'numberValue':  evs,
                    'format': "RAW"
                },
                {
                    'name': 'r2-score',
                    'numberValue':  r2s,
                    'format': "RAW"
                }
            ]
        }

        with open('/mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)
        with open('/mlpipeline-metrics.json', 'r') as f:
            print(f.read())

        print("wrote to file /mlpipeline-metrics.json")

        joblib.dump(self._clf, trained_model_path)

    def predict(self, params: list):
        trained_model_path = os.getenv('TRAINED_MODEL_PATH')

        if self._clf is None:
            if os.path.exists(trained_model_path):
                self._clf = joblib.load(trained_model_path)
            else:
                self.train()

        return self._clf.predict([params])[0]
