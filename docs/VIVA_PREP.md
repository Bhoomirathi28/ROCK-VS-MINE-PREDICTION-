# Viva preparation

Likely questions grouped by marking criterion, with short answers you can expand in your own words. Every number matches `reports/metrics/`.

## Data cleaning and splitting

**What does one row of the data represent?**
One sonar echo. A frequency-modulated pulse is bounced off an object; the echo's energy is measured in 60 frequency bands. Values are normalised to 0–1. The label says whether the object was a metal cylinder (mine) or a rock.

**Your dataset had no missing values or duplicates. What did cleaning actually do?**
We audited first and found it clean. The cleaning code (median imputation, duplicate removal, range clipping, label normalisation) still runs every time, so new data is handled the same way, and a unit test injects dirty rows to prove each rule works. Saying "it was already clean" without checking would not be justified.

**Why didn't you remove outliers?**
All 325 IQR-flagged values are inside the valid 0–1 range, so they are real strong echoes, not errors. With only 208 samples, removing them would discard real information, and models like SVM with an RBF kernel and Random Forest are fairly robust to them.

**Why stratify the split?**
So train and test keep the same 53 % mine / 47 % rock ratio. Without it, a small test set could by chance have far more of one class, making accuracy misleading.

**Why scale inside the pipeline instead of before splitting?**
If you scale the whole dataset first, the scaler's mean and standard deviation include test rows. That leaks test information into training and inflates scores. In a pipeline, the scaler is fitted on each training fold only.

**Why does Random Forest not need scaling?**
Trees split on thresholds of one feature at a time, so multiplying a feature by a constant does not change the splits.

**Why random_state = 1?**
Reproducibility, and it is the seed used in our original notebook, so our baselines match the notebook exactly.

## Model comparison and performance analytics

**Which model is best and why?**
SVM with RBF kernel (C = 5, γ = 0.01): highest cross-validated accuracy (0.855 ± 0.051), 0.857 test accuracy, best ROC-AUC (0.959), and it detected 20 of 22 mines.

**KNN has 90.5 % test accuracy. Why not deploy it?**
We select on cross-validation, not on the test set. The difference is two readings out of 42. Choosing by test score would make the test result optimistically biased. Also, tuned KNN uses k = 1: it memorises the training data and its "probabilities" are just 0 or 1, which makes confidence meaningless.

**Why is recall the most important metric here?**
Recall for mines = mines detected ÷ actual mines. A missed mine (false negative) can sink a ship; a false alarm (false positive) only costs an extra inspection. So we prefer models with high recall, and we used recall as the tie-break.

**What is ROC-AUC?**
The probability that the model gives a randomly chosen mine a higher mine-score than a randomly chosen rock. 0.5 is guessing, 1.0 is perfect. It measures ranking quality independent of the 0.5 threshold.

**What is cross-validation and why repeat it?**
The training set is split into 5 folds; the model trains on 4 and is scored on the 5th, rotating through all 5. Repeating with 10 different shuffles gives 50 scores, so we see the mean and the spread. One 42-row test split moves 2.4 points per error, which is too noisy to compare models on alone.

**Why did logistic regression perform worst?**
It draws one straight boundary. The mean signatures (Figure 2) overlap heavily and differ in non-linear ways, and neighbouring bands are strongly correlated. A non-linear model like SVM-RBF captures this.

**Tuned Random Forest scored lower on test than the baseline. Isn't tuning supposed to help?**
Tuning optimised CV accuracy on the training data. On the tiny test set the difference (0.810 vs 0.762) is two readings, within noise. This is exactly why we report CV alongside the test score.

**Several models have 100 % training accuracy. Is that overfitting?**
Training accuracy alone doesn't tell you. RF and 1-NN always fit training data perfectly by design. What matters is the generalisation estimate: CV and test scores around 0.85 show they still generalise reasonably.

**What do the C and gamma parameters of SVM do?**
C controls how much the model is penalised for misclassifying training points (high C → tighter fit, risk of overfitting). Gamma controls how far one training point's influence reaches in the RBF kernel (high gamma → very local, wiggly boundary).

**What does a confusion matrix show?**
Counts of actual vs predicted classes. For SVM on the test set: 16 rocks correct, 4 rocks flagged as mines, 2 mines missed, 20 mines caught.

## UI deployment and cross-platform accessibility

**Explain the architecture.**
Browser or any HTTP client → Flask app (served by gunicorn in production) → a `Predictor` class that loads all four saved scikit-learn pipelines once at startup → JSON response. The same API serves the web page, batch requests and any external app.

**What happens when someone enters invalid input?**
`validate_features` checks there are exactly 60 finite numbers between 0 and 1. The API returns HTTP 400 with a specific message ("Expected 60 values, received 2."), and the interface shows it next to the button.

**How is it cross-platform?**
Responsive layout tested from 390 px phones to desktop, light and dark mode, keyboard navigation, installable as a PWA on Android, iOS and desktop, a REST API with CORS for other apps, a Docker image for any OS, a CLI, and CI that runs on Ubuntu, Windows and macOS.

**Why are the model files saved in the repo?**
So the app starts immediately after cloning or deploying, without retraining. They're small (a few MB). If they're missing, the app retrains automatically on startup.

**Why do you pin library versions?**
A model pickled with one scikit-learn version may not load, or may behave differently, with another.

**What is the service worker for?**
It caches the page, styles and scripts, so the interface opens instantly and offline. Predictions still need the server.

## Code quality and GitHub

**How is the code organised?**
One module per responsibility: config, preprocessing, model definitions, evaluation, visualisation, training, prediction, and the web app. Constants live in `config.py` so the split and seed are defined once.

**How do you know it works?**
24 automated tests covering data cleaning, the split, input validation, prediction output and every API endpoint. GitHub Actions retrains from scratch and runs them on three operating systems on every push.

**How would someone reproduce your results?**
Clone, `pip install -r requirements-dev.txt`, `python -m src.train`. Same seed, same split, same numbers.

## Questions to expect about limitations

* *Is 208 samples enough?* It's small; that's why we used repeated CV and report standard deviations. More data would be the first improvement.
* *Would this work on a real ship?* Not directly: the data comes from one lab setup. It shows the approach works; a real system needs data from real conditions.
* *How could you catch more mines?* Lower the decision threshold below 0.5 for the mine class, trading more false alarms for fewer misses, or soft-vote SVM and KNN.

## Demo script (about 3 minutes)

1. Open the live URL on a laptop and on a phone. Point out the same interface on both.
2. Pick a test reading → *Classify reading*. Show the verdict, confidence, all four votes, and the "actually a mine/rock" check.
3. Switch the model to Logistic Regression on the same reading; show it can disagree.
4. Paste 59 values to show validation; then upload a CSV with several rows to show batch results.
5. Scroll to the comparison table and the CV and confusion-matrix charts; explain why SVM was deployed.
6. Show GitHub: folder structure, green CI badge, tests.
