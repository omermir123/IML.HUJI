import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda p, curr_x, curr_y: losses.append(p.loss(X, y)))
        perceptron.fit(X, y)
        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(go.Scatter(x=list(range(len(losses))), y=losses, name=n, mode="lines"),
                        layout=go.Layout(title=f"{n}",
                                         xaxis_title="x - Iterations",
                                         yaxis_title="y - training loss values",
                                         height=400))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        bayes = GaussianNaiveBayes()
        bayes.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        models = [bayes, lda]
        model_names = [fr"$\text{{Gaussian Naive Bayes with accuracy of: {accuracy(y, bayes.predict(X))}}}$",
                       fr"$\text{{LDA with accuracy of: {accuracy(y, lda.predict(X))}}}$"]
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        fig = make_subplots(rows=1, cols=2, subplot_titles=model_names, horizontal_spacing=0.1)

        # Add traces for data-points setting symbols and colors
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=m.predict(X), symbol=class_symbols[y],
                                                   colorscale=class_colors(m.classes_.size),
                                                   line=dict(color="black", width=1)))],
                           rows=1, cols=i + 1)
        # Add `X` dots specifying fitted Gaussians' means
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color="black", symbol="x"))],
                           rows=1, cols=i + 1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_traces([get_ellipse(bayes.mu_[i], np.diag(bayes.vars_[i])) for i in range(bayes.classes_.size)],
                       rows=1, cols=1)
        fig.add_traces([get_ellipse(lda.mu_[i], lda.cov_) for i in range(lda.classes_.size)],
                       rows=1, cols=2)
        fig.update_layout(title=rf"$\textbf{{{f[:-4]} Dataset}}$", width=1000, height=500, margin=dict(t=100))
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
