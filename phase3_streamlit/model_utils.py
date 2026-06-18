"""Data and model helpers for the interactive Streamlit application."""

import numpy as np
import streamlit as st
from sklearn.datasets import make_circles
from sklearn.svm import SVC


@st.cache_data(show_spinner=False)
def create_circle_dataset(
    n_samples: int,
    noise: float,
    random_state: int = 42,
):
    """Create reproducible circular classification data."""
    X, y = make_circles(
        n_samples=n_samples,
        factor=0.35,
        noise=noise,
        random_state=random_state,
    )
    return X, 1 - y


@st.cache_resource(show_spinner=False)
def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float,
    gamma: float,
    degree: int,
) -> SVC:
    """Train an SVC using the controls exposed by the application."""
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    return model.fit(X, y)


def calculate_decision_grid(
    model: SVC,
    X: np.ndarray,
    resolution: int = 220,
):
    """Return mesh coordinates and SVM decision scores."""
    padding = 0.35
    x_values = np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, resolution)
    y_values = np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)
    return xx, yy, scores
