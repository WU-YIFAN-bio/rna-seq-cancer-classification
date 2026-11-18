# RNA-seq Cancer Type Classification and Feature Selection Stability

## Project overview

This project uses gene expression measurements to classify tumor samples into five cancer types. It compares logistic regression with a linear support vector machine (SVM), examines whether fewer expression features can retain strong classification performance, and evaluates how consistently an ANOVA-based selector chooses features when the training samples change.

The project addresses three questions:

1. How well do the two classifiers distinguish the five tumor types in this dataset?
2. How does classification performance change when using 50, 200, 1,000, or all eligible features?
3. How stable is the selection of the top 200 features under repeated training-data subsampling?

This is a learning and analysis project using already-labeled tumor samples. It does not detect cancer in healthy people, predict prognosis, or provide a clinically validated diagnosis.

## Dataset, citation, and license

This project uses the **Gene Expression Cancer RNA-Seq** dataset from the **UCI Machine Learning Repository**. It contains **801 instances** and **20,531 features**, with five class labels: **BRCA, COAD, KIRC, LUAD, and PRAD**.

The two input files are `data.csv` (expression values) and `labels.csv` (sample class labels).

### Dataset citation

Fiorini, S. (2016). gene expression cancer RNA-Seq [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R88H.

### Original data acknowledgement

The original data set (hosted at https://www.synapse.org/#!Synapse:syn4301332) is maintained by the cancer genome atlas pan-cancer analysis project.

### Dataset license and processing

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license (https://creativecommons.org/licenses/by/4.0/legalcode). This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

This project loads the CSV files from the dataset cited above. After loading, the project processes and analyzes the data using sample-label alignment, train/test splitting, constant-feature removal, ANOVA feature selection, standardization, and PCA visualization of the training data, as described below. The shared results are outputs derived from these analyses.

The source dataset is licensed under CC BY 4.0. The project code is licensed under the MIT License, provided in the LICENSE file.

## Analysis methods

### 1. Input checks and sample alignment

The program reads sample IDs from the first column of both CSV files and excludes these identifiers from predictive features. It checks for empty inputs, missing or duplicate IDs, inconsistent sample sets, missing labels, non-numeric or non-finite expression values, and completely duplicated expression rows. It also checks that there are five label categories.

Labels are reordered by sample ID to match the expression matrix; matching row order is not assumed. Problems detected by these checks stop execution rather than being silently removed or imputed.

### 2. Training and final test split

A stratified 80/20 split with `random_state=42` produces **640 training samples and 161 test samples**. The final test set is excluded from model selection, PCA fitting, and feature stability analysis.

### 3. Exploratory PCA and baseline

For visualization, constant features are removed from the training data, the remaining features are standardized, and PCA produces two components. The resulting plot is exploratory; these PCA coordinates are not inputs to the classifiers.

A `DummyClassifier(strategy="most_frequent")` provides a reference score through stratified five-fold cross-validation within the training set. It predicts the most frequent class in each fold's training portion without using expression values to distinguish samples.

### 4. Model and feature-count comparison

Two classifiers are evaluated: regularized logistic regression and linear SVM (`LinearSVC`). Each uses this pipeline:

```text
Remove constant features → ANOVA top-k selection → StandardScaler → Classifier
```

ANOVA F scores rank features by between-class variation relative to within-class variation. Features are not selected at random or solely by overall variance. The selector evaluates each feature individually and does not explicitly remove redundancy between correlated features.

`GridSearchCV` evaluates:

- Feature counts: **50, 200, 1,000, and all eligible features**.
- Regularization parameter `C`: **0.1, 1.0, and 10.0**.
- Stratified five-fold cross-validation with shuffling and seed 42.
- **Macro-F1** as the selection metric, giving each class equal weight.

All 24 candidate combinations are evaluated. Constant removal, feature selection, standardization, and classification are fitted within each fold's training portion, then applied to its validation portion. A fixed feature count does not require the selected feature identities to be identical across folds.

The configuration with the highest mean validation Macro-F1 is selected. Ties retain the first encountered configuration under the current search order; runtime and stability do not automatically break ties. The selected pipeline is refitted on all training samples before final testing. Convergence warnings are treated as errors.

### 5. Feature selection stability

Using training data only, the program performs **20 stratified subsamples**, each containing 80% of the training samples (512 samples). For every subsample, it removes constant features and independently selects the **top 200 ANOVA-ranked features**.

It records each feature's selection count and frequency. It also computes Jaccard similarity for each pair of selected sets:

```text
Jaccard similarity = number of shared features / number of distinct features in either set
```

The mean over 190 pairs is included in the summary. This is a descriptive stability measure, not an independent-sample confidence interval. The analysis concerns the fixed ANOVA top-200 selector, not classifier coefficients, and does not change the selected classification model.

### 6. Final evaluation

The refitted selected pipeline predicts the reserved test samples. Outputs include Accuracy, Macro-F1, per-class Precision/Recall/F1 and sample counts, plus a confusion matrix. Final test scores are reported rather than used to select a configuration.

## Results

With the default settings, the main workflow generates the following six files in `results/`.

| File | Contents |
|---|---|
| `training_pca.png` | PCA plot of training samples, colored by class. |
| `cv_comparison.csv` | Models, feature counts, C values, cross-validation Macro-F1 means and standard deviations, and mean fitting times. |
| `feature_selection_frequency.csv` | Each feature's selection count and frequency across repeated subsamples. |
| `test_classification_report.csv` | Per-class Precision, Recall, F1, sample counts, and aggregate test metrics. |
| `test_confusion_matrix.png` | Counts of true versus predicted classes in the final test set. |
| `summary.json` | Selected configuration, split sizes, baseline and test scores, stability summary, and recorded package versions. |

## Code overview

| Code file | Responsibility |
|---|---|
| `main.py` | Configure the experiment, validate inputs, split data, coordinate analysis, and save the summary. |
| `modeling.py` | Define cross-validation, build pipelines, compare configurations, and return the refitted selected pipeline. |
| `evaluation.py` | Produce training PCA, quantify feature stability, and evaluate the final test predictions. |

## How to run

### 1. Prepare Python and dependencies

Prepare Python and the following dependencies. The versions used for this project are listed below.

| Component | Version |
|---|---|
| Python | 3.12.10 |
| NumPy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |

### 2. Place the input files

Download the dataset from the page (https://doi.org/10.24432/C5R88H), then extract the downloaded archive.

Place `data.csv` and `labels.csv` in a folder named `TCGA-PANCAN-HiSeq-801x20531` within the project directory. This folder should be in the same directory as `main.py`.

### 3. Run the program

Run the program to generate the six output files in the `results` folder. The folder is created automatically if it does not already exist.

## AI use statement

ChatGPT Work was used for code generation and documentation writing. The project author ran the code and reviewed the workflow and results.
