from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WESAD_HRV_windows_all.csv"
MODEL_PATH = BASE_DIR / "stress_model_logreg.pkl"

FEATURES = ["MeanNN", "SDNN", "RMSSD"]
LABEL_COL = "label"
SUBJ_COL = "subject"
TIME_COL = "window_start"

df = pd.read_csv(DATA_PATH).sort_values([SUBJ_COL, TIME_COL]).reset_index(drop=True)
model = joblib.load(MODEL_PATH)

X = df[FEATURES].to_numpy(dtype=float)
y = df[LABEL_COL].to_numpy(dtype=int)
p = model.predict_proba(X)[:, 1]
df["p_stress"] = p

def majority_smooth(binary_preds, k, m):
    out = []
    window = []
    for b in binary_preds:
        window.append(int(b))
        if len(window) > k:
            window.pop(0)
        out.append(1 if sum(window) >= m else 0)
    return np.array(out, dtype=int)

def apply_smoothing(df, thr, k, m):
    y_raw = (df["p_stress"].to_numpy() >= thr).astype(int)
    y_s = np.zeros(len(df), dtype=int)
    for subj, g in df.groupby(SUBJ_COL, sort=False):
        idx = g.index.to_numpy()
        y_s[idx] = majority_smooth(y_raw[idx], k=k, m=m)
    return y_s

# denenecek kombinasyonlar
thresholds = np.round(np.arange(0.35, 0.81, 0.05), 2)
km_list = [(5,3), (5,2), (7,3), (7,4)]

rows = []
for (k,m) in km_list:
    for thr in thresholds:
        y_pred = apply_smoothing(df, thr=thr, k=k, m=m)

        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        far = fp / (fp + tn)  # false alarm rate

        rows.append({
            "k": k, "m": m, "thr": thr,
            "precision": prec, "recall": rec, "f1": f1,
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
            "false_alarm_rate": float(far),
        })

res = pd.DataFrame(rows)

# hedef: recall >= 0.50 iken precision maksimum
candidates = res[res["recall"] >= 0.50].sort_values(["precision", "f1"], ascending=False)

print("\nTop 15 configs (recall>=0.50, sorted by precision then f1):")
print(candidates.head(15).to_string(index=False))

out_path = BASE_DIR / "tune_smoothing_results.csv"
res.to_csv(out_path, index=False)
print("\nSaved:", out_path)

best = candidates.head(1)
if len(best) == 1:
    b = best.iloc[0].to_dict()
    print("\nBEST (recall>=0.50):", b)
else:
    print("\nNo config met recall>=0.50")
