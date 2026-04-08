from pathlib import Path
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIG
# ============================================================
BASELINE_CSV = Path("ppg_windows_features.csv")   # Stage 3 output from calm session
TEST_CSV     = Path("ppg_windows_features2.csv")       # Stage 3 output from actual test session

MODEL_PATH   = Path("stress_model_logreg_best.pkl")
SEARCH_PATH  = Path("LOSO_model_search_results.csv")

OUT_ALL_PATH = Path("live_combined_for_inference.csv")
OUT_PRED_PATH = Path("live_predictions.csv")

SUBJECT_ID = "live_subject_01"

# majority smoothing
K = 5
M = 3

# default threshold
THR = 0.50


# ============================================================
# FEATURE SETS (must match your train/inference scripts)
# ============================================================
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


# ============================================================
# HELPERS
# ============================================================
def safe_zscore(x: pd.Series, mean: pd.Series, std: pd.Series, eps: float = 1e-6) -> pd.Series:
    mean_safe = mean.astype(float).replace([np.inf, -np.inf], np.nan)
    std_safe = std.astype(float).replace([np.inf, -np.inf], np.nan)

    mean_safe = mean_safe.fillna(mean_safe.median())
    std_safe = std_safe.fillna(std_safe.median())
    std_safe = std_safe.where(np.abs(std_safe) > eps, eps)

    return (x.astype(float) - mean_safe) / std_safe


def majority_smooth(binary_preds, k=5, m=3):
    out = []
    window = []

    for b in binary_preds:
        window.append(int(b))
        if len(window) > k:
            window.pop(0)
        out.append(1 if sum(window) >= m else 0)

    return np.array(out, dtype=int)


def ensure_required_columns(df: pd.DataFrame, name: str):
    required = [
        "subject", "phase", "label", "window_start",
        "MeanNN", "MedianNN", "IQRNN", "MADNN",
        "SDNN", "RMSSD",
        "MeanHR", "StdHR",
        "LF_power", "HF_power", "LFHF",
        "BeatDensity", "motion_std",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


# ============================================================
# 1) LOAD BASELINE + TEST
# ============================================================
if not BASELINE_CSV.exists():
    raise FileNotFoundError(f"Baseline CSV not found: {BASELINE_CSV}")

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

base_df = pd.read_csv(BASELINE_CSV)
test_df = pd.read_csv(TEST_CSV)

ensure_required_columns(base_df, "BASELINE_CSV")
ensure_required_columns(test_df, "TEST_CSV")

# force consistency
base_df = base_df.copy()
test_df = test_df.copy()

base_df["subject"] = SUBJECT_ID
test_df["subject"] = SUBJECT_ID

# strongly enforce phases
base_df["phase"] = "Base"

# keep test phase if you already set it, otherwise assign one
if "phase" not in test_df.columns or test_df["phase"].isna().all():
    test_df["phase"] = "Test"

# combine
df = pd.concat([base_df, test_df], ignore_index=True)
df = df.sort_values(["subject", "window_start"]).reset_index(drop=True)

# save combined raw table
df.to_csv(OUT_ALL_PATH, index=False)
print(f"Saved combined table: {OUT_ALL_PATH}")


# ============================================================
# 2) PICK BEST FEATURE SET FROM SEARCH RESULTS
# ============================================================
if not SEARCH_PATH.exists():
    raise FileNotFoundError(f"Search results file not found: {SEARCH_PATH}")

search_df = pd.read_csv(SEARCH_PATH)
if "feature_set" not in search_df.columns:
    raise ValueError("LOSO_model_search_results.csv missing 'feature_set' column.")

best_cfg = search_df.sort_values(
    by=["stress_f1", "roc_auc", "weighted_f1"],
    ascending=False
).iloc[0]

best_feature_set = best_cfg["feature_set"]
if best_feature_set not in FEATURE_SETS:
    raise ValueError(f"Unknown best feature set: {best_feature_set}")

best_features = FEATURE_SETS[best_feature_set]
print(f"Best feature set: {best_feature_set}")
print(f"Features: {best_features}")


# ============================================================
# 3) BASELINE NORMALIZATION (same logic as your inference code)
# ============================================================
base_feature_cols = [
    "MeanNN", "MedianNN", "IQRNN", "MADNN",
    "SDNN", "RMSSD",
    "MeanHR", "StdHR",
    "LF_power", "HF_power", "LFHF",
    "BeatDensity", "motion_std"
]

baseline_rows = df[df["phase"] == "Base"].copy()
if len(baseline_rows) == 0:
    raise RuntimeError("No rows with phase == 'Base'. Baseline normalization cannot be computed.")

base_means = (
    baseline_rows.groupby("subject")[base_feature_cols]
    .mean()
    .rename(columns={c: f"{c}_base_mean" for c in base_feature_cols})
)

base_stds = (
    baseline_rows.groupby("subject")[base_feature_cols]
    .std()
    .rename(columns={c: f"{c}_base_std" for c in base_feature_cols})
)

df = df.merge(base_means, on="subject", how="left")
df = df.merge(base_stds, on="subject", how="left")

base_stat_cols = list(base_means.columns) + list(base_stds.columns)
for c in base_stat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(df[c].median())

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

df["LF_power_log"] = np.log1p(df["LF_power"].clip(lower=0))
df["HF_power_log"] = np.log1p(df["HF_power"].clip(lower=0))
df["LFHF_log"] = np.log1p(df["LFHF"].clip(lower=0))


# ============================================================
# 4) CLEANUP FOR MODEL INPUT
# ============================================================
needed_cols = ["subject", "phase", "label", "window_start"] + best_features
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed_cols).copy()

if len(df) == 0:
    raise RuntimeError("No rows remain after cleanup. Check generated features and baseline stats.")

print(f"Rows after cleanup: {len(df)}")


# ============================================================
# 5) LOAD MODEL AND PREDICT
# ============================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

X = df[best_features].to_numpy(dtype=float)
p_stress = model.predict_proba(X)[:, 1]

df["p_stress"] = p_stress
df["y_pred_raw"] = (df["p_stress"] >= THR).astype(int)

# same smoothing logic, per subject, in time order
df = df.sort_values(["subject", "window_start"]).reset_index(drop=True)

smooth_preds = np.zeros(len(df), dtype=int)
for subj, g in df.groupby("subject", sort=False):
    idx = g.index.to_numpy()
    smooth_preds[idx] = majority_smooth(g["y_pred_raw"].to_numpy(), k=K, m=M)

df["y_pred_smooth"] = smooth_preds


# ============================================================
# 6) SAVE OUTPUTS
# ============================================================
df["best_feature_set"] = best_feature_set
df["decision_threshold"] = THR
df["smoothing_k"] = K
df["smoothing_m"] = M

df.to_csv(OUT_PRED_PATH, index=False)
print(f"Saved predictions: {OUT_PRED_PATH}")

print("\nPrediction summary:")
print(df[[
    "window_start", "phase", "p_stress", "y_pred_raw", "y_pred_smooth"
]].to_string(index=False))