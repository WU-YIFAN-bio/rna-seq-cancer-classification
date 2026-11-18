"""Entry point: fill in both CSV paths before running this file.

Required packages: numpy, pandas, scikit-learn, and matplotlib.
This program does not automatically download data, install packages, or call online services.
Verify the data license yourself and read the comments before your first run.
"""

from pathlib import Path
import json


# ==================== USER CONFIGURATION ====================
# Expression CSV: sample IDs in the first column, then numeric expression features.
# Labels CSV: sample IDs in the first column, then exactly one cancer type column.
# Both files require headers. Relative paths start from this script's directory.
EXPRESSION_CSV = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
LABELS_CSV = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
# For a smaller first run, try candidate lists such as (200,) and (1.0,).
FEATURE_COUNTS = (50, 200, 1000, "all")
C_VALUES = (0.1, 1.0, 10.0)
RUN_STABILITY = True
STABILITY_K = 200
STABILITY_REPEATS = 20
STABILITY_TRAIN_FRACTION = 0.80
# ======================================================

PROJECT_DIR = Path(__file__).resolve().parent


def configured_path(value, setting_name):
    """Stop on a blank path instead of treating it as a directory."""
    if not value.strip():
        raise ValueError(f"Set {setting_name} in main.py first; the path is currently blank.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.is_file():
        raise ValueError(f"{setting_name} points to a missing file. Check the configured path.")
    return path


def load_and_check_data(expression_path, labels_path):
    """Read only the two configured files without scanning other directories."""
    import numpy as np
    import pandas as pd

    X = pd.read_csv(expression_path, index_col=0, encoding="utf-8-sig")
    labels = pd.read_csv(labels_path, index_col=0, encoding="utf-8-sig")
    for name, table in (("Expression data", X), ("Label data", labels)):
        if table.empty or table.index.hasnans:
            raise ValueError(f"{name} is empty or contains missing sample IDs.")
        table.index = table.index.astype(str).str.strip()
        if table.index.duplicated().any() or (table.index == "").any():
            raise ValueError(f"{name} contains duplicate or blank sample IDs. Review the input first.")
    if labels.shape[1] != 1:
        raise ValueError("The labels CSV must contain exactly one label column in addition to sample IDs.")
    if set(X.index) != set(labels.index):
        raise ValueError("Expression and label sample IDs do not match exactly. Row correspondence will not be guessed.")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate expression feature names were found. Review the source file.")
    y = labels.iloc[:, 0].reindex(X.index)
    if y.isna().any():
        raise ValueError("Labels contain missing values.")
    y = y.astype(str).str.strip()
    if (y == "").any():
        raise ValueError("Labels contain blank values.")
    X = X.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Expression data contain missing or infinite values. Decide how to handle them first.")
    if X.duplicated().any():
        raise ValueError("Identical expression records were found. Review possible duplicates to prevent leakage between splits.")
    if y.nunique() != 5:
        raise ValueError("This analysis expects five classes. Check the labels file.")
    if X.shape[1] < 2:
        raise ValueError("Expression data must contain at least two features.")
    print(f"Data checks passed: {X.shape[0]} samples and {X.shape[1]} features.")
    print("Class distribution:")
    print(y.value_counts().sort_index().to_string())
    print("Check whether any patient contributes multiple samples; this initial implementation has no patient grouping metadata.")
    return X, y


def main():
    # Stop on blank paths before importing analysis packages, reading data, or training.
    expression_path = configured_path(EXPRESSION_CSV, "EXPRESSION_CSV")
    labels_path = configured_path(LABELS_CSV, "LABELS_CSV")

    import sklearn
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from modeling import make_cv, compare_models
    from evaluation import plot_training_pca, analyze_stability, evaluate_test

    X, y = load_and_check_data(expression_path, labels_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    if y_train.value_counts().min() < CV_FOLDS:
        raise ValueError("The smallest training class has too few samples for the configured number of CV folds.")

    # Save directly to a fixed directory; reruns overwrite same-named outputs.
    output = PROJECT_DIR / "results"
    output.mkdir(parents=True, exist_ok=True)
    cv = make_cv(CV_FOLDS, RANDOM_STATE)

    print("Step 1: create an exploratory PCA plot using training data only.")
    plot_training_pca(X_train, y_train, output, RANDOM_STATE)

    print("Step 2: cross-validate the majority-class baseline within the training set.")
    baseline_scores = cross_val_score(
        DummyClassifier(strategy="most_frequent"), X_train, y_train,
        scoring="f1_macro", cv=cv, n_jobs=1, error_score="raise"
    )

    print("Step 3: compare logistic regression, linear SVM, and feature counts.")
    print("Multiple fits are required and may take time. The held-out test set has not been used.")
    model, model_name, comparison = compare_models(
        X_train, y_train, cv, FEATURE_COUNTS, C_VALUES, RANDOM_STATE
    )
    comparison.to_csv(output / "cv_comparison.csv", index=False)
    print(comparison.sort_values("mean_cv_macro_f1", ascending=False).head().to_string(index=False))
    print(f"Model selected by cross-validation: {model_name}")

    if RUN_STABILITY:
        print("Step 4: analyze fixed-size feature selection stability within the training set.")
        stability = analyze_stability(
            X_train, y_train, output, STABILITY_K, STABILITY_REPEATS,
            STABILITY_TRAIN_FRACTION, RANDOM_STATE
        )
    else:
        stability = {"status": "not_run"}

    print("Step 5: evaluate the selected pipeline, refitted on all training data, on the held-out test set.")
    metrics = evaluate_test(model, X_test, y_test, output)
    summary = {
        "selected_model": model_name,
        "selected_feature_count_setting": model.named_steps["select"].k,
        "selected_C": model.named_steps["classifier"].C,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "baseline_cv_macro_f1_mean": float(baseline_scores.mean()),
        "baseline_cv_macro_f1_std": float(baseline_scores.std()),
        "stability": stability,
        "test_metrics": metrics,
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("Final test results:", metrics)
    print(f"Results saved within the project: {output.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}. Check its installation in your project environment; this program will not install it automatically.")
    except ValueError as exc:
        raise SystemExit(f"Stopped: {exc}")
