"""Preprocessing, feature selection, and training. Learned transformations stay within cross-validation."""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_cv(folds, random_state):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)


def compare_models(X, y, cv, feature_counts, c_values, random_state):
    """Compare candidates on training data and return the best pipeline refitted on all training samples."""
    if not feature_counts or not c_values:
        raise ValueError("Feature count and C candidate lists must not be empty.")
    if any(c <= 0 for c in c_values):
        raise ValueError("C must be positive.")

    # Check training-fold feature counts so k does not exceed the count after constant removal.
    available = min(
        int((X.iloc[indices].var(axis=0, ddof=0) > 0).sum())
        for indices, _ in cv.split(X, y)
    )
    if available == 0:
        raise ValueError("A training fold has no variable features available.")
    candidates = []
    for k in feature_counts:
        if k == "all" or (isinstance(k, int) and 1 <= k <= available):
            if k not in candidates:
                candidates.append(k)
        else:
            print(f"Skipping invalid feature count {k}; each training fold has at least {available} variable features.")
    if not candidates:
        raise ValueError("No valid feature count settings remain.")

    models = {
        # Use default L2 regularization to avoid explicit penalty argument changes across versions.
        "logistic_regression": LogisticRegression(solver="lbfgs", max_iter=2000),
        "linear_svm": LinearSVC(
            dual="auto", penalty="l2", loss="squared_hinge",
            max_iter=10000, random_state=random_state
        ),
    }
    tables = []
    best_search, best_name = None, None
    for name, classifier in models.items():
        print(f"Fitting: {name}")
        pipeline = Pipeline([
            ("variance", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif)),
            ("scale", StandardScaler()),
            ("classifier", classifier),
        ])
        search = GridSearchCV(
            pipeline,
            {"select__k": candidates, "classifier__C": list(c_values)},
            scoring="f1_macro", cv=cv, refit=True, n_jobs=1,
            error_score="raise", return_train_score=False,
        )
        # Treat convergence warnings as errors; review iteration limits or the analysis if needed.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(X, y)
        result = search.cv_results_
        tables.append(pd.DataFrame({
            "model": name,
            "feature_count_setting": [p["select__k"] for p in result["params"]],
            "C": [p["classifier__C"] for p in result["params"]],
            "mean_cv_macro_f1": result["mean_test_score"],
            "std_cv_macro_f1": result["std_test_score"],
            "mean_fit_seconds": result["mean_fit_time"],
        }))
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search, best_name = search, name
    return best_search.best_estimator_, best_name, pd.concat(tables, ignore_index=True)
