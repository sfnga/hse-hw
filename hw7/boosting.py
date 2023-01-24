from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

sns.set(style='darkgrid')


def create_boostrap(n_samples, subsample):
    trn_idx = np.random.choice(range(n_samples), int(n_samples * subsample))
    test_idx = list(set(range(n_samples)) - set(trn_idx))
    return trn_idx, test_idx


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
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y**2 * self.sigmoid(-y * z) * (
            1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        trn_idx, _ = create_boostrap(len(y), self.subsample)
        X_train = x[trn_idx]
        y_train = y[trn_idx]
        pred_train = predictions[trn_idx]

        model = self.base_model_class(**self.base_model_params)
        model.fit(X_train, y_train)
        new_predictions = model.predict(x)
        self.models.append(model)

        gamma = self.find_optimal_gamma(y,
                                        old_predictions=predictions,
                                        new_predictions=new_predictions)
        self.gammas.append(self.learning_rate * gamma)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        loss_increase = 0

        for _ in range(self.n_estimators):
            self.fit_new_base_model(
                x_train, -self.loss_derivative(y=y_train, z=train_predictions),
                train_predictions)
            model = self.models[-1]
            gamma = self.gammas[-1]
            trn_pred = gamma * model.predict(x_train)
            val_pred = gamma * model.predict(x_valid)
            train_predictions += trn_pred
            valid_predictions += val_pred

            train_loss = self.loss_fn(y_train, train_predictions)
            val_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(val_loss)
            self.history['train_metric'].append(self.score(x_train, y_train))
            self.history['valid_metric'].append(self.score(x_valid, y_valid))

            if self.early_stopping_rounds is not None:
                if self.history['valid_loss'][-1] >= self.history[
                        'valid_loss'][-2]:
                    loss_increase += 1
                else:
                    loss_increase = 0
                if self.early_stopping_rounds == loss_increase:
                    break
        if self.plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            ax[0].plot(np.arange(self.n_estimators),
                       self.history['train_loss'],
                       label='train')
            ax[0].plot(np.arange(self.n_estimators),
                       self.history['valid_loss'],
                       label='valid')
            ax[0].set_xlabel('n_estimators')
            ax[0].set_ylabel('loss')
            ax[0].set_title('Loss')
            ax[1].plot(np.arange(self.n_estimators),
                       self.history['train_metric'],
                       label='train')
            ax[1].plot(np.arange(self.n_estimators),
                       self.history['valid_metric'],
                       label='valid')
            ax[1].set_xlabel('n_estimators')
            ax[1].set_ylabel('metric')
            ax[1].set_title('Metric')
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros((x.shape[0], 2))
        for gamma, model in zip(self.gammas, self.models):
            predictions[:, 1] += gamma * model.predict(x)

        predictions = self.sigmoid(predictions)
        # масштабируем вероятности, roc-auc не поменяется, так как для этой метрики масштаб не важен
        predictions[:, 1] = (predictions[:, 1] - predictions[:, 1].min()) / (
            predictions[:, 1].max() - predictions[:, 1].min())
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        importances = np.zeros_like(self.models[0].feature_importances_)
        for model in self.models:
            importances += model.feature_importances_
        importances /= len(self.models)
        return importances