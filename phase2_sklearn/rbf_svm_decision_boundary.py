"""Compare linear and RBF SVM decision boundaries on circular data."""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.svm import SVC

from dataset_utils import create_circle_dataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASS_COLORS = ("#2f80ed", "#eb5757")


def create_mesh_grid(X: np.ndarray, resolution: int = 350):
    """Create a plotting grid with padding around the observed samples."""
    padding = 0.35
    x_values = np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, resolution)
    y_values = np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    return xx, yy


def plot_decision_boundary(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    annotation: str,
    output_path: Path,
) -> None:
    """Draw data, decision boundary, margins, and support vectors."""
    xx, yy = create_mesh_grid(X)
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.contourf(
        xx,
        yy,
        scores,
        levels=[scores.min(), 0, scores.max()],
        colors=CLASS_COLORS,
        alpha=0.12,
    )
    axis.contour(
        xx,
        yy,
        scores,
        levels=[-1, 0, 1],
        colors=["#777777", "#222222", "#777777"],
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 2.2, 1.2],
    )

    for label, color, name in (
        (0, CLASS_COLORS[0], "Inner class"),
        (1, CLASS_COLORS[1], "Outer class"),
    ):
        points = X[y == label]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=color,
            edgecolors="white",
            linewidths=0.5,
            s=38,
            label=name,
        )

    axis.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=115,
        facecolors="none",
        edgecolors="#f2c94c",
        linewidths=1.8,
        label="Support vectors",
    )

    axis.set(
        title=title,
        xlabel="Feature x1",
        ylabel="Feature x2",
        aspect="equal",
    )
    axis.text(
        0.02,
        0.02,
        annotation,
        transform=axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    axis.legend(loc="upper right")
    axis.grid(alpha=0.16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Train both models and save their comparison plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = create_circle_dataset()

    models = (
        (
            SVC(kernel="linear", C=1.0),
            "Linear SVM: circular data cannot be cleanly separated",
            "A linear SVM is restricted to a straight decision boundary.",
            OUTPUT_DIR / "linear_svm_decision_boundary.png",
        ),
        (
            SVC(kernel="rbf", C=1.0, gamma=1.0),
            "RBF SVM: non-linear decision boundary",
            "The RBF kernel forms a curved boundary in the original 2D space.",
            OUTPUT_DIR / "rbf_svm_decision_boundary.png",
        ),
    )

    for model, title, annotation, output_path in models:
        model.fit(X, y)
        plot_decision_boundary(model, X, y, title, annotation, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
