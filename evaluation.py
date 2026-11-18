"""Training-set exploration, feature selection stability, and final test evaluation. Figures are saved locally."""

import itertools

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a graphical window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def plot_training_pca(X, y, output, random_state):
    """Explore training data only; this separately fitted PCA is not used in model cross-validation."""
    variable = VarianceThreshold().fit_transform(X)
    if min(variable.shape) < 2:
        raise ValueError("There are too few training samples or variable features for a two-dimensional PCA plot.")
    scaled = StandardScaler().fit_transform(variable)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=random_state)
    embedding = pca.fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(y.unique()):
        mask = y.to_numpy() == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=18, alpha=0.7, label=label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Training data only: PCA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_pca.png", dpi=160)
    plt.close(fig)


def analyze_stability(X, y, output, k, repeats, fraction, random_state):
    """Keep ANOVA top-k fixed while resampling training data; never use the held-out test set."""
    if repeats < 2 or not 0 < fraction < 1 or not isinstance(k, int) or k < 1:
        raise ValueError("Stability analysis requires repeats >= 2, 0 < fraction < 1, and a positive integer k.")
    splitter = StratifiedShuffleSplit(
        n_splits=repeats, train_size=fraction, random_state=random_state
    )
    counts = np.zeros(X.shape[1], dtype=int)
    feature_sets = []
    names = np.asarray(X.columns, dtype=str)
    for indices, _ in splitter.split(X, y):
        subset = X.iloc[indices]
        variance = VarianceThreshold().fit(subset)
        available_indices = np.flatnonzero(variance.get_support())
        if len(available_indices) < k:
            raise ValueError("A stability subsample has fewer available features than STABILITY_K. Reduce that setting.")
        selector = SelectKBest(f_classif, k=k).fit(variance.transform(subset), y.iloc[indices])
        chosen = available_indices[selector.get_support()]
        counts[chosen] += 1
        feature_sets.append(set(chosen.tolist()))

    overlaps = [
        len(a & b) / len(a | b)
        for a, b in itertools.combinations(feature_sets, 2)
    ]
    frequencies = pd.DataFrame({
        "feature": names, "selection_count": counts,
        "selection_frequency": counts / repeats,
    }).sort_values(["selection_frequency", "feature"], ascending=[False, True])
    frequencies.to_csv(output / "feature_selection_frequency.csv", index=False)
    # Pairwise results are dependent; report a descriptive mean, not an independent-sample confidence interval.
    return {
        "method": "ANOVA top-k selection under stratified subsampling",
        "k": k, "repeats": repeats, "training_fraction": fraction,
        "mean_pairwise_jaccard": float(np.mean(overlaps)),
    }


def evaluate_test(model, X_test, y_test, output):
    """Use the final test set for reporting only, not for further model selection."""
    predicted = model.predict(X_test)
    labels = model.classes_
    report = classification_report(
        y_test, predicted, labels=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output / "test_classification_report.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=labels, display_labels=labels,
        ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title("Held-out test set")
    fig.tight_layout()
    fig.savefig(output / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, labels=labels, average="macro", zero_division=0)),
    }
