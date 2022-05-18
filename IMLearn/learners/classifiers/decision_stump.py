from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        feature_to_sign = np.array(list(product(range(X.shape[1]), [-1, 1])))
        all_thresholds = np.array([self._find_threshold(X[:, f], y, s) for f, s in feature_to_sign])
        temp = np.argmin(all_thresholds[:, 1])
        self.j_ = feature_to_sign[temp, 0]
        self.sign_ = feature_to_sign[temp, 1]
        self.threshold_ = all_thresholds[temp, 0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        X_y_sorted = np.column_stack((values, labels))
        X_y_sorted = X_y_sorted[X_y_sorted[:, 0].argsort()]
        temp = np.array([sign] * values.size)
        all_poss = np.zeros((values.size, 2))
        for i in range(values.size):
            temp_labels = X_y_sorted[:, 1] * temp
            all_poss[i] = np.array([X_y_sorted[i, 0], np.abs(np.sum(temp_labels[temp_labels < 0]))])
            temp[i] = -sign
        return all_poss[np.argmin(all_poss[:, 1])]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        D = np.abs(y)
        true_y = np.sign(y)
        return np.sum(D[true_y != self.predict(X)])
