from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable

import numpy
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = numpy.zeros(cv)
    validation_score = numpy.zeros(cv)
    folds_X = np.array_split(X, cv)
    folds_y = np.array_split(y, cv)
    for i in range(cv):
        X_train = np.concatenate(folds_X[:i] + folds_X[i+1:], axis=0)
        y_train = np.concatenate(folds_y[:i] + folds_y[i+1:], axis=0)
        estimator.fit(X_train, y_train)
        validation_score[i] = scoring(estimator.predict(folds_X[i]), folds_y[i])
        train_score[i] = scoring(estimator.predict(X_train), y_train)
    return np.mean(train_score), np.mean(validation_score)


