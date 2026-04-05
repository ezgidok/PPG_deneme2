from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import time

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# ------------------------------------------------------------
# LOG helper
# ------------------------------------------------------------
def mark(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------
# Paths / Ayarlar
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "WESAD_HRV_windows_all.csv"
MODEL_PATH = BASE_DIR / "stress_model_logreg_best.pkl"
SEARCH_PATH = BASE_DIR / "LOSO_model_search_results.csv"
OUT_PATH = BASE_DIR / "WESAD_predictions_with_majority_smoothing.csv"

LABEL_COL = "label"
SUBJ_COL = "subject"
TIME_COL = "window_start"
PHASE_COL = "phase"

# Majority smoothing
K = 5
M = 3

# threshold
# İstersen bunu daha sonra best threshold scan çıktısından da otomatik çekebiliriz.
THR = 0.50


# ------------------------------------------------------------
# Train dosyası ile aynı FEATURE_SETS
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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


def summarize(name, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

    auc_val = roc_auc_score(y_true, y_prob)
    print("ROC-AUC:", round(float(auc_val), 4))


# ------------------------------------------------------------
# 1) Dosya kontrolleri
# ------------------------------------------------------------
mark(f"DATA_PATH exists? {DATA_PATH.exists()} -> {DATA_PATH}")
mark(f"MODEL_PATH exists? {MODEL_PATH.exists()} -> {MODEL_PATH}")
mark(f"SEARCH_PATH exists? {SEARCH_PATH.exists()} -> {SEARCH_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV bulunamadı: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

if not SEARCH_PATH.exists():
    raise FileNotFoundError(f"Model search sonuç dosyası bulunamadı: {SEARCH_PATH}")


# ------------------------------------------------------------
# 2) CSV oku
# ------------------------------------------------------------
mark("Reading CSV...")
df = pd.read_csv(DATA_PATH, engine="c", low_memory=False)
mark(f"CSV loaded: shape={df.shape}")

required_raw = [
    "subject", "phase", "label", "window_start",
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
# 3) En iyi feature set’i search sonuçlarından çek
# ------------------------------------------------------------
mark("Reading model search results...")
search_df = pd.read_csv(SEARCH_PATH)

if "feature_set" not in search_df.columns:
    raise ValueError("LOSO_model_search_results.csv içinde 'feature_set' kolonu yok.")

best_cfg = search_df.sort_values(
    by=["stress_f1", "roc_auc", "weighted_f1"],
    ascending=False
).iloc[0]

best_feature_set = best_cfg["feature_set"]

if best_feature_set not in FEATURE_SETS:
    raise ValueError(f"Tanımsız best feature set: {best_feature_set}")

best_features = FEATURE_SETS[best_feature_set]
mark(f"Best feature set: {best_feature_set} -> {best_features}")


# ------------------------------------------------------------
# 4) Train script ile aynı baseline normalization
# ------------------------------------------------------------
BASE_PHASE_NAMES = {"Base"}

base_feature_cols = [
    "MeanNN", "MedianNN", "IQRNN", "MADNN",
    "SDNN", "RMSSD",
    "MeanHR", "StdHR",
    "LF_power", "HF_power", "LFHF",
    "BeatDensity", "motion_std"
]

mark("Computing subject baseline stats...")
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

base_stat_cols = list(base_means.columns) + list(base_stds.columns)
for c in base_stat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(df[c].median())

mark("Generating normalized features...")
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


# ------------------------------------------------------------
# 5) Cleanup
# ------------------------------------------------------------
needed_cols = [LABEL_COL, SUBJ_COL, TIME_COL, PHASE_COL] + best_features
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed_cols).copy()
mark(f"After cleanup: shape={df.shape}")

y_true = df[LABEL_COL].astype(int).to_numpy()
X = df[best_features].to_numpy(dtype=float)

mark(f"Final X shape: {X.shape}")
mark(f"Label counts: {pd.Series(y_true).value_counts().to_dict()}")


# ------------------------------------------------------------
# 6) Model yükle ve predict et
# ------------------------------------------------------------
mark("Loading model...")
model = joblib.load(MODEL_PATH)
mark("Model loaded.")

mark("Predicting probabilities...")
p_stress = model.predict_proba(X)[:, 1]
df["p_stress"] = p_stress
df["y_pred_raw"] = (df["p_stress"] >= THR).astype(int)
mark("Raw prediction done.")


# ------------------------------------------------------------
# 7) Majority smoothing
# ------------------------------------------------------------
mark("Applying majority smoothing per subject...")
df = df.sort_values([SUBJ_COL, TIME_COL]).reset_index(drop=True)

smooth_preds = np.zeros(len(df), dtype=int)

for subj, g in df.groupby(SUBJ_COL, sort=False):
    idx = g.index.to_numpy()
    smooth_preds[idx] = majority_smooth(
        g["y_pred_raw"].to_numpy(),
        k=K,
        m=M
    )

df["y_pred_smooth"] = smooth_preds
mark("Smoothing done.")


# ------------------------------------------------------------
# 8) Evaluation
# ------------------------------------------------------------
summarize(
    name=f"RAW (thr={THR})",
    y_true=y_true,
    y_pred=df["y_pred_raw"].to_numpy(),
    y_prob=df["p_stress"].to_numpy()
)

summarize(
    name=f"SMOOTH majority K={K}, M={M} (thr={THR})",
    y_true=y_true,
    y_pred=df["y_pred_smooth"].to_numpy(),
    y_prob=df["p_stress"].to_numpy()
)

cm_raw = confusion_matrix(y_true, df["y_pred_raw"].to_numpy())
cm_s = confusion_matrix(y_true, df["y_pred_smooth"].to_numpy())

tn_r, fp_r, fn_r, tp_r = cm_raw.ravel()
tn_s, fp_s, fn_s, tp_s = cm_s.ravel()

print("\n--- FP/FN change ---")
print(f"RAW:    FP={fp_r}, FN={fn_r}, TP={tp_r}, TN={tn_r}")
print(f"SMOOTH: FP={fp_s}, FN={fn_s}, TP={tp_s}, TN={tn_s}")


# ------------------------------------------------------------
# 9) CSV kaydet
# ------------------------------------------------------------
extra_info = {
    "best_feature_set": best_feature_set,
    "decision_threshold": THR,
    "smoothing_k": K,
    "smoothing_m": M,
}
for k, v in extra_info.items():
    df[k] = v

df.to_csv(OUT_PATH, index=False)
mark(f"Saved detailed predictions: {OUT_PATH}")