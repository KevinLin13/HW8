"""Dataset helpers for the static SVM demonstration."""

from sklearn.datasets import make_circles


def create_circle_dataset(
    n_samples: int = 300,
    factor: float = 0.35,
    noise: float = 0.06,
    random_state: int = 42,
):
    """Return a reproducible inner-circle versus outer-circle dataset."""
    X, y = make_circles(
        n_samples=n_samples,
        factor=factor,
        noise=noise,
        random_state=random_state,
    )
    return X, 1 - y
