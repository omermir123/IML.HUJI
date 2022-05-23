import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy



def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    loss_train = np.array([adaboost.partial_loss(train_X, train_y, t + 1) for t in range(n_learners)])
    loss_test = np.array([adaboost.partial_loss(test_X, test_y, t + 1) for t in range(n_learners)])
    x = np.linspace(1, n_learners, n_learners)
    go.Figure([
        go.Scatter(x=x, y=loss_train, mode='lines', name=rf"$\text{{Loss on Train}}$"),
        go.Scatter(x=x, y=loss_test, mode='lines', name=rf"$\text{{Loss on Test}}$")]).update_layout(
        title=rf"$\text{{Loss of train and test as a function od iteration in Adaboost}}$").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    titles = [rf"$\text{{the decision boundary obtained after {t} iterations}}$" for t in T]
    fig_Q2 = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.03)
    for i, t in enumerate(T):
        fig_Q2.add_traces(

            [decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y.astype(int), symbol=class_symbols[test_y.astype(int)],
                                    colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig_Q2.update_layout(title=rf"$\textbf{{Decision Boundaries After 5, 50, 100, 250 Iterations}}$",
                         margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_Q2.show()
    # Question 3: Decision surface of best performing ensemble
    errors = np.array([adaboost.partial_loss(test_X, test_y, t + 1) for t in range(n_learners)])
    best_t = np.argmin(errors) + 1
    go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, best_t), lims[0], lims[1], showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=test_y.astype(int), symbol=class_symbols[test_y.astype(int)],
                                      colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1)))]).update_layout(
        title=rf"$\textbf{{The ensemble size achieved the lowest test error is {best_t} with accuracy of "
              rf"{accuracy(test_y, adaboost.partial_predict(test_X, best_t))}}}$", width=1000, height=500).update_xaxes(
        visible=False).update_yaxes(visible=False).show()
    # Question 4: Decision surface with weighted samples
    D = (adaboost.D_ / np.max(adaboost.D_)) * 5
    go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y.astype(int), symbol="circle",
                                      colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1), size=D))]).update_layout(
        title=rf"$\textbf{{The Training set with a point size proportional to it’s weight}}$", width=1000,
        height=500).update_xaxes(visible=False).update_yaxes(visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
