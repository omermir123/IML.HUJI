import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import mean_square_error, misclassification_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go
from sklearn.metrics import roc_curve


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights, values = [], []

    def callback(weight, value, **kwargs):
        weights.append(weight)
        values.append(value)

    return callback, weights, values


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    models = [L1, L2]
    models_name = ["L1", "L2"]
    for i, model in enumerate(models):
        convergence_rate_fig = go.Figure(data=[], layout=go.Layout(
            title=f"The Convergence rate of {models_name[i]} model for all etas"))
        losses = []
        for eta in etas:
            callback, weights, values = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            losses.append(model(gd.fit(f=model(np.copy(init)), X=None, y=None)).compute_output())
            plot_descent_path(module=model, descent_path=np.array(weights),
                              title=rf"$\text{{{models_name[i]} with }}\eta={eta} $").show()
            convergence_rate_fig.add_traces(
                [go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="markers", name=rf"$\eta={eta}$")])
        print(f"The best loss for {models_name[i]} is {min(losses)} with eta={etas[np.argmin(losses)]}")
        convergence_rate_fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure(data=[], layout=go.Layout(title=f"The L1 convergence rate"))
    losses = []
    for gamma in gammas:
        callback, weights, values = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback)
        losses.append(L1(gd.fit(f=L1(np.copy(init)), X=None, y=None)).compute_output())
        fig.add_traces(
            [go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="markers", name=rf"$\gamma={gamma}$")])
    fig.show()
    print(f"The best loss for L1 is {min(losses)} with eta=0.1 and gamma={gammas[np.argmin(losses)]}")

    # Plot algorithm's convergence for the different values of gamma
    # raise NotImplementedError()

    # Plot descent path for gamma=0.95
    callback, weights, values = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=callback)
    gd.fit(f=L1(np.copy(init)), X=None, y=None)
    plot_descent_path(module=L1, descent_path=np.array(weights),
                      title=rf"$\text{{L1 with }} \eta={eta} \text{{ and }} \gamma=0.95$").show()
    callback, weights, values = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=callback)
    gd.fit(f=L2(np.copy(init)), X=None, y=None)
    plot_descent_path(module=L2, descent_path=np.array(weights),
                      title=rf"$\text{{L2 with }} \eta={eta} \text{{ and }} \gamma=0.95$").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # callback, weights, values = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=gd)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_train)
    # Plotting convergence rate of logistic regression over SA heart disease data
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    # thresholds = np.round(thresholds, 2)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model}}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    alpha_star = thresholds[np.argmax(tpr - fpr)]
    model = LogisticRegression(solver=gd, alpha=alpha_star)
    model.fit(X_train, y_train)
    print(
        f"the alpha* is {round(alpha_star, 2)} and the test error for that alpha is {round(model.loss(X_test, y_test), 3)}")
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    models = ["l1", "l2"]
    lam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for model in models:
        res = np.zeros((len(lam), 2))
        for i in range(len(lam)):
            res[i] = np.array(
                cross_validate(LogisticRegression(solver=gd, alpha=0.5, penalty=model, lam=lam[i]), X_train, y_train,
                               misclassification_error, cv=5))
        lambda_star = lam[np.argmin(res[:, 1])]
        logistic = LogisticRegression(solver=gd, alpha=0.5, penalty=model, lam=lambda_star)
        logistic.fit(X_train, y_train)
        print(
            f"the best lambda for {model}-regularized logistic regression is {lambda_star} "
            f"with test error of {logistic.loss(X_test, y_test)}")



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
