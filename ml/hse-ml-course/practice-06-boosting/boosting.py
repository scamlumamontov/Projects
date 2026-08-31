from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(1 + self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        model = DecisionTreeRegressor(**self.base_model_params)
        rem = -self.loss_derivative(y, predictions)
        model.fit(x, rem)

        new_predictions = model.predict(x)
        best_gamma = self.find_optimal_gamma(y, predictions, new_predictions)
        self.gammas.append(best_gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.models[-1].predict(x_train) * self.gammas[-1] * self.learning_rate
            valid_predictions += self.models[-1].predict(x_valid) * self.gammas[-1] * self.learning_rate

            if self.early_stopping_rounds is not None:
                pass

        if self.plot:
            pass

    def predict_proba(self, x):
        sum = 0
        for gamma, model in zip(self.gammas, self.models):
            sum += model.predict(x) * gamma * self.learning_rate
        p = self.sigmoid(sum)
        p = np.stack([(1 - p), p], axis=1)
        return p

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        sum = None
        for model in self.models:
            if sum is None:
                sum = model.feature_importances_
            else:
                sum += model.feature_importances_
        total_sum = sum.sum()
        imp = dict( zip( [f"f_{i}" for i in np.arange(0, len(sum), 1).astype(int)], sum / total_sum))
        imp = dict( sorted(imp.items(), key=lambda x : -x[1]) )
        return imp