from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    label_names,
    save_path: str | Path | None = None,
):
    """
    Plot and optionally save the confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=label_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(
        "Emotion Classification - Confusion Matrix"
    )

    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("Actual Emotion")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_class_distribution(
    y_true,
    label_names,
    save_path: str | Path | None = None,
):
    """
    Plot the distribution of classes in the dataset.
    """

    counts = np.bincount(
        y_true,
        minlength=len(label_names),
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        label_names,
        counts,
    )

    ax.set_title(
        "Emotion Class Distribution"
    )

    ax.set_xlabel("Emotion")
    ax.set_ylabel("Number of Samples")

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    for index, count in enumerate(counts):
        ax.text(
            index,
            count,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()