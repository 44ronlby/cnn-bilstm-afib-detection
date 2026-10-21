
import os
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = "outputs/confusion_matrix.png",
    labels: list = ["Normal", "AFib"]
) -> None:
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(
        "Confusion Matrix - AFib Detection",
        fontsize=12,
        fontweight="bold"
    )
    plt.ylabel("Actual Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to '{save_path}'")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = "outputs/roc_curve.png"
) -> float:
    """Generates and saves the ROC Curve plot and returns the calculated AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"CNN-BiLSTM (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.title(
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=12,
        fontweight="bold"
    )
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"ROC curve saved to '{save_path}'")

    plt.close()

    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = "outputs/pr_curve.png"
) -> None:
    """Generates and saves the Precision-Recall Curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        color="purple",
        lw=2,
        label="Precision-Recall Curve"
    )

    plt.xlabel("Recall (Sensitivity)", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.title(
        "Precision-Recall Curve",
        fontsize=12,
        fontweight="bold"
    )
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Precision-Recall curve saved to '{save_path}'")

    plt.close()


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = "outputs/confidence_distribution.png"
) -> None:
    """Plots histogram/KDE distribution of prediction probabilities separated by ground-truth class."""
    plt.figure(figsize=(8, 5))

    sns.histplot(
        y_probs[y_true == 0],
        color="blue",
        label="Normal (Class 0)",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.5
    )
    sns.histplot(
        y_probs[y_true == 1],
        color="red",
        label="AFib (Class 1)",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.5
    )

    plt.xlabel("Predicted AFib Probability", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.title(
        "Model Confidence Distribution",
        fontsize=12,
        fontweight="bold"
    )
    plt.legend(loc="upper center")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Confidence distribution saved to '{save_path}'")

    plt.close()


def plot_training_history(
    history: Any,
    save_path: Optional[str] = "outputs/training_history.png"
) -> None:
    """Plots loss and accuracy training vs. validation curves across epochs."""
    hist = history.history if hasattr(history, "history") else history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(
        hist["loss"],
        label="Train Loss",
        color="blue",
        lw=2
    )

    if "val_loss" in hist:
        ax1.plot(
            hist["val_loss"],
            label="Val Loss",
            color="orange",
            lw=2
        )

    ax1.set_title("Model Loss Progression", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Crossentropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy Curve
    acc_key = "accuracy" if "accuracy" in hist else "acc"

    ax2.plot(
        hist[acc_key],
        label="Train Accuracy",
        color="blue",
        lw=2
    )

    if "val_accuracy" in hist or "val_acc" in hist:
        val_acc_key = (
            "val_accuracy"
            if "val_accuracy" in hist
            else "val_acc"
        )
        ax2.plot(
            hist[val_acc_key],
            label="Val Accuracy",
            color="orange",
            lw=2
        )

    ax2.set_title("Model Accuracy Progression", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Training history saved to '{save_path}'")

    plt.close()


def evaluate_and_plot_all(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "outputs/",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Runs evaluation on test set and exports all visual diagnostic charts."""
    os.makedirs(output_dir, exist_ok=True)

    y_probs = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "AFib"]
        )
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    roc_auc = plot_roc_curve(
        y_test,
        y_probs,
        save_path=os.path.join(output_dir, "roc_curve.png")
    )

    plot_precision_recall_curve(
        y_test,
        y_probs,
        save_path=os.path.join(output_dir, "pr_curve.png")
    )

    plot_confidence_distribution(
        y_test,
        y_probs,
        save_path=os.path.join(output_dir, "confidence_distribution.png")
    )

    return {
        "y_probs": y_probs,
        "y_pred": y_pred,
        "auc": roc_auc
    }
