import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import os
import joblib


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

        joblib.dump(self._clf, trained_model_path)

    def predict(self, params: list):
        trained_model_path = os.getenv('TRAINED_MODEL_PATH')

        if self._clf is None:
            if os.path.exists(trained_model_path):
                self._clf = joblib.load(trained_model_path)
            else:
                self.train()

        return self._clf.predict([params])[0]