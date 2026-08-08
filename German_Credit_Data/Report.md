# German Credit Binary Classification Report

## 1. Project understanding

This experiment predicts whether one credit applicant will have a bad credit outcome. UCI raw class `2` is the positive bad-credit class and class `1` is the good-credit class. The intended use is an offline benchmark or decision-support prototype, not an autonomous lending decision. Because business error costs were not supplied, model ranking uses average precision and the operating threshold uses training-only F2, which weights recall of bad credit more heavily than precision.

## 2. Data requirements

The source is [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), fetched through `ucimlrepo` as dataset 144. Each row is one application, with 20 application-time predictors and one outcome. The extract contains 1000 rows. The dataset does not include timestamps, repeated-customer identifiers, extraction history, or production lineage, so an IID stratified split is assumed. Licensing, retention, consent, and lawful-use requirements must be confirmed before any operational use.

## 3. Data analysis and findings

The raw extract has 0 missing cells and 0 exact duplicate rows. It contains 700 good-credit and 300 bad-credit cases, so bad-credit prevalence is 30.0%. This 70/30 split is moderately imbalanced, not an extreme rare-event problem. Numeric variables have materially different scales, and categorical variables are coded rather than ordinal measurements. Detailed profiles are in `feature_profile.csv` and `numeric_summary.csv`; plots are in `figures/`.

## 4. Cleaning and preprocessing

The cryptic UCI attributes were renamed to semantic fields. Numeric values are median-imputed and standardized; categoricals are most-frequent-imputed and one-hot encoded with unknown-category handling. Although this snapshot has no missing values, imputers make the saved inference pipeline robust to permitted missing input. All learned transformations are fitted inside cross-validation folds. Raw target 1 maps to 0 (good), and raw target 2 maps to 1 (bad).

## 5. Feature engineering and selection

Two prediction-time-safe ratios were added: credit amount per duration month and credit amount per existing credit. An L1-regularized logistic selector is inside the pipeline and its threshold was tuned by GridSearchCV. GridSearchCV tested stronger pruning, but the best validation result retained all 63 encoded features; the aggressive `1.25*mean` threshold reduced the random forest's best mean validation average precision from 0.658 to 0.613. This is an evidence-based no-deletion result, not a claim that every feature is equally important. Holdout permutation importance ranks the strongest original associations as: checking_account_status, credit_amount_per_month, purpose, credit_amount, credit_history, other_debtors_guarantors, age_years, telephone. These are predictive associations, not causal effects.

## 6. Model development experiments

The benchmark is a stratified dummy classifier. Tuned candidates are logistic regression (interpretable linear baseline), random forest (bagged nonlinear interactions), and histogram gradient boosting (boosted nonlinear interactions). The bounded grids cover regularization, tree complexity, selection threshold, and class weighting. GridSearchCV selected class weighting for the final model. Synthetic oversampling was not used because the imbalance is moderate and SMOTE-style interpolation is questionable for one-hot-coded mixed categorical data.

## 7. Training and validation approach

The split is stratified and reproducible: 800 training rows and an untouched 200-row holdout. GridSearchCV uses five shuffled stratified folds on training data only and refits by mean average precision. Average precision directly evaluates retrieval of the 30% bad-credit class; ROC-AUC is secondary. The final threshold of 0.266 was selected from out-of-fold training probabilities by maximum F2, never from the holdout.

## 8. Evaluation results

| model | cv_average_precision_mean | cv_average_precision_std | test_average_precision | test_roc_auc | test_recall_bad_at_0_5 |
|---|---|---|---|---|---|
| Dummy prevalence baseline | 0.300 | 0.000 | 0.300 | 0.500 | 0.000 |
| Logistic Regression | 0.609 | 0.083 | 0.626 | 0.793 | 0.783 |
| Random Forest | 0.658 | 0.080 | 0.670 | 0.809 | 0.700 |
| Histogram Gradient Boosting | 0.608 | 0.056 | 0.628 | 0.790 | 0.533 |

The selected model is **Random Forest**, chosen only by cross-validated average precision. On the untouched test set, average precision is 0.670 (bootstrap 95% CI 0.551-0.781) and ROC-AUC is 0.809 (95% CI 0.743-0.872). At threshold 0.266, bad-credit precision is 0.378, recall is 0.933, F2 is 0.722, with 92 false positives and 4 false negatives. The uncertainty is substantial because the holdout has only 200 records.

## 9. Interpretation and error analysis

Permutation importance was calculated once on the untouched holdout after model selection and is therefore descriptive, not a new selection step. The confusion matrix and ROC/precision-recall curves are in `figures/`. A descriptive audit was produced for 8 test-set groups. Small groups and the dataset's coded, incomplete protected-attribute representation prevent a formal fairness conclusion. `personal_status_sex`, age, and foreign-worker status are protected characteristics or proxies in many lending contexts; aggregate accuracy cannot establish non-discrimination.

## 10. Recommendation, deployment considerations, limitations, and next steps

Use the saved pipeline only as a reproducible research baseline. Do not deploy it for automated credit decisions without: contemporary representative data; a documented decision policy and error costs; legal and fairness review; probability calibration on new data; subgroup sample-size requirements; human-review and adverse-action processes; and external/temporal validation. A proposed batch deployment should validate schema and ranges, version inputs and model artifacts, log outcomes, monitor missingness/category drift/prediction drift and subgroup errors, and roll back when data contracts fail. Retrain only after labels arrive and predefined degradation thresholds are breached.

The largest limitations are the small 1,000-row historical sample, absence of timestamps and geographic context, coded and incomplete protected attributes, possible dataset obsolescence, and a single random holdout. The next highest-value experiment is repeated nested cross-validation or external validation on recent applicants, followed by explicit cost-sensitive threshold selection and calibration using operational requirements.

## 11. Reproducibility and artifacts

- Data snapshot: `data/german_credit.csv`
- Complete raw-input inference pipeline: `artifacts/model.joblib`
- Reload and raw-row scoring check: `artifacts/inference_smoke_test.json`
- Metrics and parameters: `artifacts/metrics.json`, `artifacts/model_comparison.csv`
- Feature and error analysis: `artifacts/selected_transformed_features.csv`, `artifacts/permutation_importance.csv`, `artifacts/subgroup_metrics.csv`
- Figures: `artifacts/figures/`
- Random seed: 42; split: 80/20 stratified; GridSearchCV: 5 folds
- Environment: Python 3.14.0, pandas 3.0.5, NumPy 2.5.1, scikit-learn 1.9.0
- Reproduce from the repository root: `.env/bin/python German_Credit_Data/main.py`

The model artifact has not been tested in a production serving environment.
