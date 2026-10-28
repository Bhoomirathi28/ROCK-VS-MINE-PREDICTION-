"""All report figures, drawn in the project's marine palette."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src import config  # noqa: E402
from src.evaluate import mine_probability  # noqa: E402
from src.models import MODEL_LABELS  # noqa: E402

ABYSS = "#0E2A3B"
DEEP = "#1F4E6B"
SONAR = "#3FA7C9"
FOAM = "#9CD4E6"
SLATE = "#5B6B78"
FOG = "#C9D3DA"
MODEL_COLOURS = {"logistic_regression": SLATE, "svm": SONAR,
                 "random_forest": DEEP, "knn": FOAM}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": SLATE, "axes.labelcolor": ABYSS, "axes.titleweight": "bold",
    "axes.titlecolor": ABYSS, "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E8EC", "grid.linewidth": 0.7,
})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name)
    plt.close(fig)


def plot_class_distribution(df):
    counts = df[config.TARGET].value_counts().reindex(["M", "R"])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["Mine (M)", "Rock (R)"], counts.values, color=[SONAR, SLATE], width=0.55)
    ax.bar_label(bars, labels=[f"{v}  ({v / counts.sum():.0%})" for v in counts.values], padding=3)
    ax.set_ylabel("Samples")
    ax.set_title("Class balance (208 sonar returns)")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, "01_class_distribution.png")


def plot_mean_signature(df):
    bands = np.arange(1, config.N_FEATURES + 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for label, colour in (("M", SONAR), ("R", SLATE)):
        part = df[df[config.TARGET] == label][config.FEATURE_NAMES]
        mean, std = part.mean().values, part.std().values
        ax.plot(bands, mean, color=colour, lw=2, label=config.CLASS_NAMES[label])
        ax.fill_between(bands, mean - std, mean + std, color=colour, alpha=0.15)
    ax.set_xlabel("Frequency band (low → high)")
    ax.set_ylabel("Returned energy")
    ax.set_title("Mean sonar signature per class (±1 std)")
    ax.legend(frameon=False)
    _save(fig, "02_mean_spectral_signature.png")


def plot_correlation(df):
    corr = df[config.FEATURE_NAMES].corr().values
    fig, ax = plt.subplots(figsize=(6, 5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sea", [SLATE, "#F2F5F7", DEEP])
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title("Correlation between the 60 frequency bands")
    ax.set_xlabel("Band"); ax.set_ylabel("Band"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "03_feature_correlation.png")


def plot_feature_boxplots(df):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.boxplot([df[c] for c in config.FEATURE_NAMES], widths=0.6, showfliers=True,
               patch_artist=True,
               boxprops=dict(facecolor=FOAM, edgecolor=DEEP),
               medianprops=dict(color=ABYSS),
               flierprops=dict(marker=".", markersize=3, markeredgecolor=SLATE))
    ax.set_xticks(range(5, 61, 5)); ax.set_xticklabels(range(5, 61, 5))
    ax.set_xlabel("Frequency band"); ax.set_ylabel("Energy")
    ax.set_title("Spread and IQR outliers per band (all values within [0, 1])")
    _save(fig, "04_feature_boxplots.png")


def plot_confusion_matrices(results, prefix="tuned"):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    fig.subplots_adjust(wspace=0.5)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cm", ["#F2F5F7", DEEP])
    for ax, (key, res) in zip(axes, results.items()):
        cm = np.array(res["confusion_matrix"])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.sum() / 2)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13,
                    color="white" if v > cm.sum() / 4 else ABYSS)
        ax.set_xticks([0, 1], ["Rock", "Mine"]); ax.set_yticks([0, 1], ["Rock", "Mine"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
        ax.set_title(f"{MODEL_LABELS[key]}\nacc {res['test_accuracy']:.3f}", fontsize=9.5)
    _save(fig, f"05_confusion_matrices_{prefix}.png")


def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    y_bin = (np.asarray(y_test) == "M").astype(int)
    for key, model in models.items():
        fpr, tpr, _ = roc_curve(y_bin, mine_probability(model, X_test))
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLOURS[key],
                label=f"{MODEL_LABELS[key]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=FOG)
    ax.set_xlabel("False alarm rate (rock flagged as mine)")
    ax.set_ylabel("Mine detection rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, "06_roc_curves.png")


def plot_metric_comparison(df_metrics):
    metrics = ["test_accuracy", "precision", "recall", "f1", "roc_auc"]
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    tuned = df_metrics[df_metrics["variant"] == "tuned"].set_index("model")
    x = np.arange(len(metrics)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(tuned.index):
        vals = tuned.loc[key, metrics].values.astype(float)
        ax.bar(x + (i - 1.5) * w, vals, w, color=MODEL_COLOURS[key], label=MODEL_LABELS[key])
    ax.set_xticks(x, names); ax.set_ylim(0.5, 1.02); ax.set_ylabel("Score")
    ax.set_title("Tuned models on the hold-out test set (Mine = positive class)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, "07_model_comparison.png")


def plot_baseline_vs_tuned(df_metrics):
    pivot = df_metrics.pivot(index="model", columns="variant", values="test_accuracy")
    order = list(MODEL_LABELS)
    pivot = pivot.reindex(order)
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    b1 = ax.bar(x - w / 2, pivot["baseline"], w, color=FOG, label="Baseline (notebook)")
    b2 = ax.bar(x + w / 2, pivot["tuned"], w, color=DEEP, label="Tuned pipeline")
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, [MODEL_LABELS[k] for k in order], fontsize=8.5)
    ax.set_ylim(0.5, 1.05); ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of scaling + hyper-parameter tuning")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "08_baseline_vs_tuned.png")


def plot_cv_boxplot(cv_scores):
    keys = list(cv_scores)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bp = ax.boxplot([cv_scores[k] for k in keys], patch_artist=True, widths=0.5,
                    medianprops=dict(color=ABYSS, lw=1.5))
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor(MODEL_COLOURS[k]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(keys) + 1), [MODEL_LABELS[k] for k in keys], fontsize=8.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("10 × 5-fold cross-validation on the training set (50 folds)")
    _save(fig, "09_cross_validation.png")


def plot_overfitting(df_metrics):
    tuned = df_metrics[df_metrics["variant"] == "tuned"].set_index("model").reindex(list(MODEL_LABELS))
    x = np.arange(len(tuned)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - w / 2, tuned["train_accuracy"], w, color=SLATE, label="Train")
    ax.bar(x + w / 2, tuned["test_accuracy"], w, color=SONAR, label="Test")
    ax.set_xticks(x, [MODEL_LABELS[k] for k in tuned.index], fontsize=8.5)
    ax.set_ylim(0.5, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Train vs test accuracy (gap = overfitting)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "10_train_vs_test.png")


def plot_feature_importance(rf_model):
    imp = rf_model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(np.arange(1, config.N_FEATURES + 1), imp, color=DEEP)
    top = np.argsort(imp)[-5:]
    ax.bar(top + 1, imp[top], color=SONAR)
    ax.set_xlabel("Frequency band"); ax.set_ylabel("Importance")
    ax.set_title("Random Forest feature importance (top 5 bands highlighted)")
    _save(fig, "11_feature_importance.png")
