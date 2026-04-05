from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "WESAD_predictions_with_majority_smoothing.csv"


# ------------------------------------------------------------
# 1) CSV oku
# ------------------------------------------------------------
print(f"Reading: {CSV_PATH}")
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("CSV loaded:", df.shape)


# ------------------------------------------------------------
# 2) Gerekli kolonları kontrol et
# ------------------------------------------------------------
required_cols = ["label", "p_stress"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolonlar var: {missing}")

has_raw = "y_pred_raw" in df.columns
has_smooth = "y_pred_smooth" in df.columns

if not has_raw and not has_smooth:
    raise ValueError("CSV içinde ne 'y_pred_raw' ne de 'y_pred_smooth' bulundu.")

y_true = df["label"].astype(int).to_numpy()
y_prob = df["p_stress"].astype(float).to_numpy()


# ------------------------------------------------------------
# 3) ROC Curve
# ------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PPG Stress Detection")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(BASE_DIR / "fig_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 4) Confusion Matrix
# ------------------------------------------------------------
def save_confusion_matrix(y_pred: np.ndarray, title_suffix: str, out_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-stress (0)", "Stress (1)"]
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {title_suffix}")
    plt.tight_layout()
    plt.savefig(BASE_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()


if has_raw:
    y_pred_raw = df["y_pred_raw"].astype(int).to_numpy()
    save_confusion_matrix(
        y_pred=y_pred_raw,
        title_suffix="RAW Predictions",
        out_name="fig_confusion_raw.png"
    )

if has_smooth:
    y_pred_smooth = df["y_pred_smooth"].astype(int).to_numpy()
    save_confusion_matrix(
        y_pred=y_pred_smooth,
        title_suffix="Majority Smoothed Predictions",
        out_name="fig_confusion_smooth.png"
    )


# ------------------------------------------------------------
# 5) Probability histogram
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label="True Non-stress (0)")
plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label="True Stress (1)")
plt.xlabel("Predicted Stress Probability")
plt.ylabel("Count")
plt.title("Predicted Probability Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "fig_probability_histogram.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 6) Threshold scan figure
# ------------------------------------------------------------
thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

f1_scores = []
prec_scores = []
rec_scores = []
acc_scores = []

for thr in thresholds:
    y_pred_thr = (y_prob >= thr).astype(int)
    f1_scores.append(f1_score(y_true, y_pred_thr, zero_division=0))
    prec_scores.append(precision_score(y_true, y_pred_thr, zero_division=0))
    rec_scores.append(recall_score(y_true, y_pred_thr, zero_division=0))
    acc_scores.append(accuracy_score(y_true, y_pred_thr))

best_idx = int(np.argmax(f1_scores))
best_thr = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])

plt.figure(figsize=(7, 5))
plt.plot(thresholds, f1_scores, marker="o", label="F1-score")
plt.plot(thresholds, prec_scores, marker="s", label="Precision")
plt.plot(thresholds, rec_scores, marker="^", label="Recall")
plt.plot(thresholds, acc_scores, marker="d", label="Accuracy")
plt.axvline(best_thr, linestyle="--", label=f"Best F1 threshold = {best_thr:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Scan")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "fig_threshold_scan.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 7) Subject-wise probability plot
# ------------------------------------------------------------
if "subject" in df.columns and "window_start" in df.columns:
    plot_df = df.sort_values(["subject", "window_start"]).copy()

    subjects = plot_df["subject"].astype(str).unique().tolist()
    n_subj = len(subjects)

    plt.figure(figsize=(12, max(6, n_subj * 1.2)))

    y_offset = 0
    yticks = []
    yticklabels = []

    for subj in subjects:
        g = plot_df[plot_df["subject"].astype(str) == subj].copy()
        x = np.arange(len(g))
        y = g["p_stress"].to_numpy()

        plt.plot(x, y + y_offset, marker="o", linewidth=1)

        if "label" in g.columns:
            stress_idx = np.where(g["label"].to_numpy() == 1)[0]
            if len(stress_idx) > 0:
                plt.scatter(stress_idx, y[stress_idx] + y_offset, marker="x")

        yticks.append(y_offset + 0.5)
        yticklabels.append(subj)
        y_offset += 1.5

    plt.xlabel("Window Index (within subject)")
    plt.ylabel("Subject")
    plt.yticks(yticks, yticklabels)
    plt.title("Predicted Stress Probabilities by Subject")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_subjectwise_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# 8) Özet çıktı
# ------------------------------------------------------------
print("\nSaved figures:")
print("- fig_roc_curve.png")

if has_raw:
    print("- fig_confusion_raw.png")
if has_smooth:
    print("- fig_confusion_smooth.png")

print("- fig_probability_histogram.png")
print("- fig_threshold_scan.png")

if "subject" in df.columns and "window_start" in df.columns:
    print("- fig_subjectwise_probabilities.png")

print("\nSummary:")
print(f"ROC AUC = {roc_auc:.4f}")
print(f"Best threshold by F1 = {best_thr:.2f}")
print(f"Best F1 = {best_f1:.4f}")