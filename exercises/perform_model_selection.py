from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    polynom = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y = polynom(X) + np.random.normal(0, noise, n_samples)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X, columns=['x']), pd.Series(y, name='y'), 2 / 3)
    X_train, y_train, X_test, y_test = X_train.values.ravel(), y_train.values, X_test.values.ravel(), y_test.values
    go.Figure([
        go.Scatter(x=X, y=polynom(X), mode='markers', name=rf"$\text{{Noiseless Model}}$"),
        go.Scatter(x=X_train, y=y_train, mode='markers', name=rf"$\text{{Train set}}$"),
        go.Scatter(x=X_test, y=y_test, mode='markers', name=rf"$\text{{Test set}}$")]).update_layout(
        title=rf"$\text{{The true model and noise model}}$").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.array(list(range(11)))
    res = np.zeros((len(degrees), 2))
    for k in degrees:
        res[k] = np.array(cross_validate(PolynomialFitting(k), X_train, y_train, mean_square_error, cv=5))
    go.Figure([
        go.Scatter(x=degrees, y=res[:, 0], mode='markers+lines', name=rf"$\text{{Average Train Loss}}$"),
        go.Scatter(x=degrees, y=res[:, 1], mode='markers+lines',
                   name=rf"$\text{{Average validation Loss}}$")]).update_layout(
        title=rf"$\text{{Train and Validation error of CV  as function of polynomial degree}}$").show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(res[:, 1])
    polyfit = PolynomialFitting(k_star)
    polyfit.fit(X_train, y_train)
    print(f"The best value of k is: {k_star}\nAnd the test error is: {round(polyfit.loss(X_test, y_test), 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = X[:n_samples, :], y[:n_samples], X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam = np.linspace(0.001, 2.5, n_evaluations)
    res_rigde = np.zeros((len(lam), 2))
    res_lasso = np.zeros((len(lam), 2))
    for i in range(n_evaluations):
        res_rigde[i] = np.array(cross_validate(RidgeRegression(lam[i]), X_train, y_train, mean_square_error, cv=5))
        res_lasso[i] = np.array(cross_validate(Lasso(lam[i]), X_train, y_train, mean_square_error, cv=5))
    titles = [rf"$\text{{Lasso Regression}}$", rf"$\text{{Ridge Regression}}$"]
    fig_Q6 = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.03)
    fig_Q6.add_traces(
        [go.Scatter(x=lam, y=res_lasso[:, 0], mode='markers+lines', name=rf"$\text{{Average Train Loss}}$"),
         go.Scatter(x=lam, y=res_lasso[:, 1], mode='markers+lines',
                    name=rf"$\text{{Average validation Loss}}$")], rows=1, cols=1)
    fig_Q6.add_traces(
        [go.Scatter(x=lam, y=res_rigde[:, 0], mode='markers+lines', name=rf"$\text{{Average Train Loss}}$"),
         go.Scatter(x=lam, y=res_rigde[:, 1], mode='markers+lines',
                    name=rf"$\text{{Average validation Loss}}$")], rows=1, cols=2)
    fig_Q6.update_layout(
        title=rf"$\text{{Train and Validation error for Lasso and ridge regression as function of lambda value}}$").show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    rigde_lambda = lam[np.argmin(res_rigde[:, 1])]
    lasso_lambda = lam[np.argmin(res_lasso[:, 1])]
    lasso_estimator = Lasso(lasso_lambda)
    lasso_estimator.fit(X_train, y_train)
    ridge_estimator = RidgeRegression(rigde_lambda)
    ridge_estimator.fit(X_train, y_train)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    print(
        f"The test loss of Lasso estimator with lambda={lasso_lambda} is "
        f"{mean_square_error(lasso_estimator.predict(X_test), y_test)}")
    print(f"The test loss of Ridge estimator with lambda={rigde_lambda} is {ridge_estimator.loss(X_test, y_test)}")
    print(f"The test loss of Linear Regression is {linear_regression.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
