# 10-page report: structure and content

A page-by-page plan for the report, mapped to the four marking criteria. Every number below comes from `reports/metrics/` and every figure from `reports/figures/`, so the report, the repository and the demo all agree. Rewrite the notes in your own words.

Suggested format: A4, 12 pt body, 1.15 spacing, figures 12–15 cm wide with numbered captions.

---

## Page 1: Title page

Project title, course (Machine Learning Lab, A5, 5th semester), team members with roll numbers, guide's name, institute, date, GitHub link and live app link.

## Page 2: Abstract, contents, introduction

**Abstract (about 150 words).** Problem, dataset, four models, best result (SVM: 85.5 % CV accuracy, 85.7 % test accuracy, 0.959 ROC-AUC, 20 of 22 mines detected), deployment.

**Introduction.** Naval mines are a hazard to shipping and divers. Sonar sends sound pulses and records the echo; a metal cylinder and a rock of similar size return subtly different frequency patterns. Manual interpretation is slow and error-prone, so an automated classifier helps. State the objective: compare four classical ML models and deploy the best as an accessible web tool.

## Page 3: Dataset and exploratory analysis

* Source and citation (Gorman & Sejnowski, 1988, UCI).
* 208 samples, 60 continuous features (energy per frequency band, 0–1), binary label: 111 mines, 97 rocks.
* Figure 1: `01_class_distribution.png`. Point: 53/47, roughly balanced, so accuracy is a fair headline metric, but stratification is still used.
* Figure 2: `02_mean_spectral_signature.png`. Point: mines return more energy in bands 1–20 and 40–48; the classes overlap heavily (wide ±1 std bands), which is why this is a hard problem and why non-linear models win.
* Figure 3: `03_feature_correlation.png`. Point: neighbouring bands are strongly correlated (diagonal block structure), which hurts plain logistic regression and motivates regularisation.

## Page 4: Data cleaning and splitting  *(criterion 1: 3 marks)*

Present the audit as a before/after table from `data_audit.json`:

| Check | Before | After | Rule applied |
|---|---|---|---|
| Missing values | 0 | 0 | Median imputation per column |
| Duplicate rows | 0 | 0 | Drop exact duplicates (prevents train/test leakage) |
| Values outside 0–1 | 0 | 0 | Clip to [0, 1] |
| Invalid / messy labels | 0 | 0 | Strip, upper-case, drop if not R/M |
| IQR outlier cells | 325 | 325 | Kept on purpose (see below) |

Key justifications to write:

* The dataset turned out to be clean, but the pipeline was built and **unit-tested with injected dirty data** (`tests/test_preprocessing.py`), so it is ready for new recordings.
* Outliers are kept: every value is inside the physically valid range, they are strong genuine echoes, and removing them from only 208 samples would cost real signal. Figure 4: `04_feature_boxplots.png`.
* Scaling is **not** done before splitting. `StandardScaler` sits inside each model's pipeline, so it learns mean and standard deviation from training folds only. Scaling the whole dataset first would leak test-set statistics into training.
* Split: 80/20, stratified, `random_state = 1` → 166 train (89 M / 77 R), 42 test (22 M / 20 R). Same seed as the notebook, so the baseline results are reproducible.
* Why also cross-validation: with 42 test rows, each misclassification moves accuracy by 2.4 points. 10 × 5-fold repeated CV gives 50 estimates per model.

## Pages 5–6: Models and methodology  *(criterion 2: 4 marks)*

One short paragraph per model: how it works, why it suits this data, what was tuned.

| Model | Idea | Scaling | Search space | Best |
|---|---|---|---|---|
| Logistic Regression | Linear decision boundary, sigmoid output | Yes | C ∈ {0.01 … 100} | C = 1 |
| SVM | Maximum-margin boundary; RBF kernel maps to a non-linear space | Yes | C ∈ {0.5 … 100}, γ ∈ {scale, 0.05, 0.01, 0.005} | C = 5, γ = 0.01 |
| Random Forest | Majority vote of bootstrapped decision trees | No (trees are scale-invariant) | trees, depth, max_features, min_samples_leaf | 500 trees, depth unlimited, sqrt features |
| KNN | Label of the nearest training readings | Yes (distance-based) | k ∈ {1 … 9}, weights, p (Manhattan/Euclidean) | k = 1, Manhattan |

Describe the protocol: baseline (notebook settings) → pipeline + `GridSearchCV` (5-fold, accuracy) → repeated CV → single final test evaluation. Model selection by CV mean, tie-break on mine recall.

Metric definitions, with Mine as the positive class: precision, recall (detection rate), specificity, F1, ROC-AUC. Explain why recall matters most: a missed mine (false negative) can cost lives; a false alarm costs a second look.

## Pages 7–8: Results and performance analytics  *(criterion 2 continued)*

* Main table (copy from README "Results").
* Figure 5: `07_model_comparison.png`, all metrics side by side.
* Figure 6: `05_confusion_matrices_tuned.png`. SVM misses 2 mines; LR misses 9.
* Figure 7: `06_roc_curves.png`. SVM has the highest AUC (0.959).
* Figure 8: `09_cross_validation.png`. SVM and KNN lead (≈ 0.85); Random Forest has the widest spread (± 0.071).
* Figure 9: `08_baseline_vs_tuned.png`. Scaling + tuning lifted SVM from 0.786 to 0.857 and KNN from 0.810 to 0.905.
* Figure 10: `10_train_vs_test.png`. Overfitting discussion: RF, SVM and 1-NN reach 100 % training accuracy; the CV score is what tells us how they generalise.
* Figure 11: `11_feature_importance.png`. Bands 9–12 and 45–49 carry the most information, which matches where the class signatures in Figure 2 separate.

Discussion points:

1. **Why SVM was deployed over KNN.** KNN won on the test set by two readings, but SVM has the higher and more stable CV score, the best AUC and smooth probabilities. Choosing on the test set would make the test score optimistically biased.
2. **Why logistic regression underperforms.** The classes overlap non-linearly (Figure 2); a single straight boundary cannot separate them. Tuning did not help (0.690 → 0.667, within noise).
3. **Why tuned Random Forest dropped on the test set** (0.810 → 0.762) while its CV score is decent: the test set is small; two readings flip. Shows why one split is not enough.
4. Inference speed: all models predict in under 1 ms per reading; RF is slowest (500 trees).

## Page 9: UI deployment and cross-platform accessibility  *(criterion 3: 4 marks)*

* Architecture diagram: Browser / phone / script → HTTPS → Flask (gunicorn) → `Predictor` → 4 joblib pipelines.
* Screenshots: `docs/screenshots/desktop.png`, `desktop-sand.png`, `mobile.png`.
* Features: test samples with known labels, paste or CSV upload (batch), model switcher, all-model consensus, echo profile against class averages, comparison table and charts.
* Input validation with clear error messages (wrong count, non-numeric, outside 0–1).
* Cross-platform: responsive down to 320 px, keyboard accessible, installable PWA (Android, iOS, desktop), REST API with CORS, Docker image, CLI, CI on three operating systems.
* Hosting: Render (live URL), plus Docker and LAN options.

## Page 10: Code quality, conclusion, future work, references  *(criterion 4: 4 marks)*

* Repository structure (tree from README); modules with single responsibilities; config in one place; docstrings.
* 24 automated tests (data, prediction, API) and GitHub Actions CI on Ubuntu, Windows and macOS; flake8 linting; pinned dependencies; MIT license.
* **Conclusion.** SVM with RBF kernel gives the best balance: 85.5 % CV, 85.7 % test, 91 % of mines detected.
* **Limitations.** 208 samples is small; data from one experimental setup; test estimates have wide confidence intervals.
* **Future work.** More data, threshold tuning to push mine recall higher, ensembles (soft voting of SVM + KNN), gradient boosting, SHAP explanations.
* **References.** Gorman & Sejnowski (1988); scikit-learn (Pedregosa et al., 2011, JMLR 12); UCI ML Repository; Flask documentation.

---

### Checklist before submitting

- [ ] Live URL works on a phone
- [ ] GitHub README shows the screenshot and results table
- [ ] Every figure in the report has a number and caption, and is referenced in the text
- [ ] Numbers in the report match `reports/metrics/model_comparison.csv`
- [ ] Each member can explain one of: cleaning, models, deployment
