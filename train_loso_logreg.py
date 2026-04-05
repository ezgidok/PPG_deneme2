from pathlib import Path
import time
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# LOG helper
# ------------------------------------------------------------
def mark(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------
# 0) Paths / Ayarlar
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WESAD_HRV_windows_all.csv"

MODEL_OUT = BASE_DIR / "stress_model_logreg_best.pkl"
SEARCH_OUT = BASE_DIR / "LOSO_model_search_results.csv"
THR_SCAN_OUT = BASE_DIR / "LOSO_threshold_scan_best.csv"
PER_SUBJ_OUT = BASE_DIR / "LOSO_logreg_best_per_subject.csv"

LABEL_COL = "label"
GROUP_COL = "subject"

# İlk kontrollü deneme için daha makul feature setleri
FEATURE_SETS = {
    "base4_z": [
        "MeanNN_z", "SDNN_z", "RMSSD_z", "motion_z"
    ],
    "robust7_z": [
        "MeanNN_z", "MedianNN_z", "SDNN_z", "RMSSD_z",
        "IQRNN_z", "MeanHR_z", "motion_z"
    ],
    "robust9_z": [
        "MeanNN_z", "MedianNN_z", "SDNN_z", "RMSSD_z",
        "IQRNN_z", "MADNN_z", "MeanHR_z", "StdHR_z", "motion_z"
    ],
    "freq10_z": [
        "MeanNN_z", "MedianNN_z", "SDNN_z", "RMSSD_z",
        "IQRNN_z", "MADNN_z", "MeanHR_z", "StdHR_z",
        "LFHF_log", "motion_z"
    ],
    "full12_z": [
        "MeanNN_z", "MedianNN_z", "SDNN_z", "RMSSD_z",
        "IQRNN_z", "MADNN_z", "MeanHR_z", "StdHR_z",
        "LF_power_log", "HF_power_log", "LFHF_log", "motion_z"
    ],
}

C_VALUES = [0.01, 0.1, 1.0, 10.0]
SOLVERS = ["liblinear", "lbfgs"]


# ------------------------------------------------------------
# 1) Veri yükleme
# ------------------------------------------------------------
mark(f"Reading CSV: {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV bulunamadı: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, engine="c", low_memory=False)
mark(f"CSV loaded: shape={df.shape}")

required_raw = [
    "subject", "phase", "label",
    "MeanNN", "MedianNN", "IQRNN", "MADNN",
    "SDNN", "RMSSD",
    "MeanHR", "StdHR",
    "LF_power", "HF_power", "LFHF",
    "BeatDensity", "motion_std"
]
missing_raw = [c for c in required_raw if c not in df.columns]
if missing_raw:
    raise ValueError(f"CSV içinde eksik kolonlar var: {missing_raw}")


# ------------------------------------------------------------
# 2) Subject baseline normalization
# ------------------------------------------------------------
# ------------------------------------------------------------
# 2) Subject baseline normalization
# ------------------------------------------------------------
BASE_PHASE_NAMES = {"Base"}

base_feature_cols = [
    "MeanNN", "MedianNN", "IQRNN", "MADNN",
    "SDNN", "RMSSD",
    "MeanHR", "StdHR",
    "LF_power", "HF_power", "LFHF",
    "BeatDensity", "motion_std"
]

base_df = df[df["phase"].isin(BASE_PHASE_NAMES)].copy()

base_means = (
    base_df.groupby("subject")[base_feature_cols]
    .mean()
    .rename(columns={c: f"{c}_base_mean" for c in base_feature_cols})
)

base_stds = (
    base_df.groupby("subject")[base_feature_cols]
    .std()
    .rename(columns={c: f"{c}_base_std" for c in base_feature_cols})
)

df = df.merge(base_means, on="subject", how="left")
df = df.merge(base_stds, on="subject", how="left")


def safe_zscore(x: pd.Series, mean: pd.Series, std: pd.Series, eps: float = 1e-6) -> pd.Series:
    mean_safe = mean.astype(float).replace([np.inf, -np.inf], np.nan)
    std_safe = std.astype(float).replace([np.inf, -np.inf], np.nan)

    mean_safe = mean_safe.fillna(mean_safe.median())
    std_safe = std_safe.fillna(std_safe.median())
    std_safe = std_safe.where(np.abs(std_safe) > eps, eps)

    return (x.astype(float) - mean_safe) / std_safe


# Base mean/std kolonlarını güvenli hale getir
base_stat_cols = list(base_means.columns) + list(base_stds.columns)
for c in base_stat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(df[c].median())

# z-score normalized features
df["MeanNN_z"] = safe_zscore(df["MeanNN"], df["MeanNN_base_mean"], df["MeanNN_base_std"])
df["MedianNN_z"] = safe_zscore(df["MedianNN"], df["MedianNN_base_mean"], df["MedianNN_base_std"])
df["IQRNN_z"] = safe_zscore(df["IQRNN"], df["IQRNN_base_mean"], df["IQRNN_base_std"])
df["MADNN_z"] = safe_zscore(df["MADNN"], df["MADNN_base_mean"], df["MADNN_base_std"])
df["SDNN_z"] = safe_zscore(df["SDNN"], df["SDNN_base_mean"], df["SDNN_base_std"])
df["RMSSD_z"] = safe_zscore(df["RMSSD"], df["RMSSD_base_mean"], df["RMSSD_base_std"])
df["MeanHR_z"] = safe_zscore(df["MeanHR"], df["MeanHR_base_mean"], df["MeanHR_base_std"])
df["StdHR_z"] = safe_zscore(df["StdHR"], df["StdHR_base_mean"], df["StdHR_base_std"])
df["BeatDensity_z"] = safe_zscore(df["BeatDensity"], df["BeatDensity_base_mean"], df["BeatDensity_base_std"])
df["motion_z"] = safe_zscore(df["motion_std"], df["motion_std_base_mean"], df["motion_std_base_std"])

# Frequency features: raw-stable transform
df["LF_power_log"] = np.log1p(df["LF_power"].clip(lower=0))
df["HF_power_log"] = np.log1p(df["HF_power"].clip(lower=0))
df["LFHF_log"] = np.log1p(df["LFHF"].clip(lower=0))
# ------------------------------------------------------------
# Safe normalization helpers
# ------------------------------------------------------------
def safe_ratio(num: pd.Series, den: pd.Series, eps: float = 1e-6) -> pd.Series:
    den_safe = den.copy().astype(float)
    den_safe = den_safe.fillna(den_safe.median())
    den_safe = den_safe.where(np.abs(den_safe) > eps, eps)
    return num.astype(float) / den_safe

# Base kolonlarını güvenli hale getir
for c in base_means.columns:
    df[c] = df[c].astype(float)
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(df[c].median())

# Relative features (time-domain / stable features)
# z-score normalized features
df["MeanNN_z"] = safe_zscore(df["MeanNN"], df["MeanNN_base_mean"], df["MeanNN_base_std"])
df["MedianNN_z"] = safe_zscore(df["MedianNN"], df["MedianNN_base_mean"], df["MedianNN_base_std"])
df["IQRNN_z"] = safe_zscore(df["IQRNN"], df["IQRNN_base_mean"], df["IQRNN_base_std"])
df["MADNN_z"] = safe_zscore(df["MADNN"], df["MADNN_base_mean"], df["MADNN_base_std"])
df["SDNN_z"] = safe_zscore(df["SDNN"], df["SDNN_base_mean"], df["SDNN_base_std"])
df["RMSSD_z"] = safe_zscore(df["RMSSD"], df["RMSSD_base_mean"], df["RMSSD_base_std"])
df["MeanHR_z"] = safe_zscore(df["MeanHR"], df["MeanHR_base_mean"], df["MeanHR_base_std"])
df["StdHR_z"] = safe_zscore(df["StdHR"], df["StdHR_base_mean"], df["StdHR_base_std"])
df["BeatDensity_z"] = safe_zscore(df["BeatDensity"], df["BeatDensity_base_mean"], df["BeatDensity_base_std"])
df["motion_z"] = safe_zscore(df["motion_std"], df["motion_std_base_mean"], df["motion_std_base_std"])

# Frequency-domain features raw-stable transform
df["LF_power_log"] = np.log1p(df["LF_power"].clip(lower=0))
df["HF_power_log"] = np.log1p(df["HF_power"].clip(lower=0))
df["LFHF_log"] = np.log1p(df["LFHF"].clip(lower=0))

# Frequency-domain features:
# ratio yerine log1p kullanmak daha stabil
df["LF_power_log"] = np.log1p(df["LF_power"].clip(lower=0))
df["HF_power_log"] = np.log1p(df["HF_power"].clip(lower=0))
df["LFHF_log"] = np.log1p(df["LFHF"].clip(lower=0))

# cleanup
all_needed_cols = {LABEL_COL, GROUP_COL}
for feat_cols in FEATURE_SETS.values():
    all_needed_cols.update(feat_cols)

missing = [c for c in all_needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolonlar: {missing}")

print("\n--- Dataset shape before cleanup ---")
print(df.shape)

rel_debug_cols = sorted(list(all_needed_cols - {LABEL_COL, GROUP_COL}))
print("\n--- Missing values per feature before dropna ---")
print(df[rel_debug_cols].isna().sum().sort_values(ascending=False).head(20))

print("\n--- Inf values per feature before dropna ---")
inf_counts = pd.Series({
    c: np.isinf(df[c].to_numpy(dtype=float)).sum()
    for c in rel_debug_cols
})
print(inf_counts.sort_values(ascending=False).head(20))

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=list(all_needed_cols)).copy()
print("\n--- Dataset shape after cleanup ---")
print(df.shape)

y = df[LABEL_COL].to_numpy(dtype=int)
groups = df[GROUP_COL].astype(str).to_numpy()

mark(f"Prepared y/groups: y={y.shape}, subjects={pd.Series(groups).nunique()}")
mark(f"Label counts: {pd.Series(y).value_counts().to_dict()}")

logo = LeaveOneGroupOut()


# ------------------------------------------------------------
# 3) Helpers
# ------------------------------------------------------------
def build_model(solver: str, c_val: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver=solver,
                C=c_val,
                random_state=42,
            )),
        ]
    )


def evaluate_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: Pipeline):
    per_subject = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        subj = groups[test_idx][0]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        auc = np.nan
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_prob)

        per_subject.append({
            "subject": subj,
            "n": int(len(y_test)),
            "acc": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "auc": float(auc) if not np.isnan(auc) else np.nan,
            "test_label_counts": dict(pd.Series(y_test).value_counts()),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

    res = pd.DataFrame(per_subject)

    all_y_true = np.array(all_y_true, dtype=int)
    all_y_pred = np.array(all_y_pred, dtype=int)
    all_y_prob = np.array(all_y_prob, dtype=float)

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()

    overall = {
        "accuracy": accuracy_score(all_y_true, all_y_pred),
        "weighted_f1": f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0),
        "stress_f1": f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0),
        "precision_1": precision_score(all_y_true, all_y_pred, pos_label=1, zero_division=0),
        "recall_1": recall_score(all_y_true, all_y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(all_y_true, all_y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "subject_acc_mean": float(res["acc"].mean()),
        "subject_acc_std": float(res["acc"].std(ddof=1)),
        "subject_f1_mean": float(res["f1"].mean()),
        "subject_f1_std": float(res["f1"].std(ddof=1)),
        "subject_auc_mean": float(res["auc"].dropna().mean()),
        "subject_auc_std": float(res["auc"].dropna().std(ddof=1)),
    }

    return res, all_y_true, all_y_pred, all_y_prob, overall


def metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "thr": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


# ------------------------------------------------------------
# 4) Model search
# ------------------------------------------------------------
mark("Starting LOSO model search...")
search_rows = []

for feat_name, feat_cols in FEATURE_SETS.items():
    X_curr = df[feat_cols].to_numpy(dtype=float)
    mark(f"Testing feature set: {feat_name} -> {feat_cols}")

    for solver in SOLVERS:
        for c_val in C_VALUES:
            mark(f"Evaluating config: feature_set={feat_name}, solver={solver}, C={c_val}")

            model = build_model(solver=solver, c_val=c_val)
            res, all_y_true, all_y_pred, all_y_prob, overall = evaluate_loso(
                X=X_curr, y=y, groups=groups, model=model
            )

            row = {
                "feature_set": feat_name,
                "features": ", ".join(feat_cols),
                "solver": solver,
                "C": c_val,
                **overall
            }
            search_rows.append(row)

search_df = pd.DataFrame(search_rows)
search_df = search_df.sort_values(
    by=["stress_f1", "roc_auc", "weighted_f1"],
    ascending=False
).reset_index(drop=True)

search_df.to_csv(SEARCH_OUT, index=False)
mark(f"Saved model search results: {SEARCH_OUT}")

print("\n--- TOP MODEL CONFIGS ---")
print(search_df.head(10).to_string(index=False))

best_cfg = search_df.iloc[0].to_dict()
print("\n--- BEST CONFIG ---")
print(best_cfg)

best_feature_set = best_cfg["feature_set"]
best_solver = best_cfg["solver"]
best_c = float(best_cfg["C"])
best_features = FEATURE_SETS[best_feature_set]


# ------------------------------------------------------------
# 5) Best config ile final LOSO evaluation
# ------------------------------------------------------------
mark("Running final LOSO evaluation with best config...")
X_best = df[best_features].to_numpy(dtype=float)
best_model = build_model(solver=best_solver, c_val=best_c)

res, all_y_true, all_y_pred, all_y_prob, overall = evaluate_loso(
    X=X_best, y=y, groups=groups, model=best_model
)

print("\n--- LOSO per-subject summary (all) ---")
print(res.to_string(index=False))

print("\n--- LOSO overall (best config) ---")
print("Best feature set:", best_feature_set, "->", best_features)
print("Best solver:", best_solver)
print("Best C:", best_c)

print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(all_y_true, all_y_pred))

print("\nClassification report:")
print(classification_report(all_y_true, all_y_pred, digits=3))

print("Overall accuracy:", round(float(overall["accuracy"]), 4))
print("Overall weighted F1:", round(float(overall["weighted_f1"]), 4))
print("Overall stress F1:", round(float(overall["stress_f1"]), 4))
print("Overall ROC-AUC:", round(float(overall["roc_auc"]), 4))
print("Per-subject mean accuracy ± std:",
      round(float(overall["subject_acc_mean"]), 4), "±", round(float(overall["subject_acc_std"]), 4))
print("Per-subject mean F1 ± std:",
      round(float(overall["subject_f1_mean"]), 4), "±", round(float(overall["subject_f1_std"]), 4))
print("Per-subject mean AUC ± std:",
      round(float(overall["subject_auc_mean"]), 4), "±", round(float(overall["subject_auc_std"]), 4))

res.to_csv(PER_SUBJ_OUT, index=False)
mark(f"Saved per-subject results: {PER_SUBJ_OUT}")


# ------------------------------------------------------------
# 6) Threshold scan
# ------------------------------------------------------------
mark("Running threshold scan for best config...")
thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
scan = [metrics_at_threshold(all_y_true, all_y_prob, t) for t in thresholds]
scan_df = pd.DataFrame(scan)

print("\n--- Threshold scan (selected columns) ---")
print(scan_df[["thr", "precision", "recall", "f1", "fp", "fn", "false_alarm_rate"]].to_string(index=False))

base_row = scan_df.loc[scan_df["thr"].sub(0.50).abs().idxmin()]
print("\nBaseline thr=0.50 metrics:")
print(base_row.to_dict())

candidates = scan_df[scan_df["recall"] >= 0.50]
if len(candidates) == 0:
    best_thr_row = scan_df.loc[scan_df["f1"].idxmax()]
    rule = "max_f1"
else:
    best_thr_row = candidates.loc[candidates["precision"].idxmax()]
    rule = "max_precision_with_recall>=0.50"

print(f"\nSelected threshold rule: {rule}")
print("Selected threshold metrics:")
print(best_thr_row.to_dict())

scan_df.to_csv(THR_SCAN_OUT, index=False)
mark(f"Saved threshold scan: {THR_SCAN_OUT}")


# ------------------------------------------------------------
# 7) Final model fit on ALL data
# ------------------------------------------------------------
mark("Fitting final model on ALL data with best config...")
best_model.fit(X_best, y)
mark("Final fit done. Saving model atomically...")

tmp_path = BASE_DIR / (MODEL_OUT.name + ".tmp")
joblib.dump(best_model, tmp_path, compress=3)
os.replace(tmp_path, MODEL_OUT)

mark(f"Saved final model: {MODEL_OUT}")
mark(f"Model size (bytes): {MODEL_OUT.stat().st_size}")
mark("Done.")