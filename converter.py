from pathlib import Path
import numpy as np
import pandas as pd

# Optional but strongly recommended for LF/HF
try:
    from scipy.signal import welch
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# CONFIG
# ============================================================
RAW_PATH = Path("ppg_stream_raw2.csv")
OUT_PATH = Path("ppg_windows_features2.csv")

SUBJECT_ID = "live_subject_01"
PHASE_NAME = "Base2"      # use "Base" for calm baseline capture first
LABEL_VALUE = 0          # dummy for now; inference can ignore ground truth if desired

WINDOW_SEC = 60.0        # window length
STEP_SEC = 5.0           # sliding step
MIN_RR_COUNT = 20        # minimum accepted RR intervals in a window

# RR acceptance gates
RR_MIN_MS = 300.0        # 200 BPM
RR_MAX_MS = 2000.0       # 30 BPM
MAX_RR_JUMP_FRAC = 0.25  # reject if jump >25% from previous accepted RR


# ============================================================
# HELPERS
# ============================================================
def median_abs_deviation(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def compute_rr_from_beats(df: pd.DataFrame) -> pd.DataFrame:
    beat_df = df[(df["beat"] == 1) & (df["finger"] == 1)].copy()
    beat_df = beat_df.sort_values("time_ms").reset_index(drop=True)

    times = beat_df["time_ms"].to_numpy(dtype=float)
    if len(times) < 2:
        return pd.DataFrame(columns=["time_ms", "rr_ms"])

    rr_times = []
    rr_vals = []

    prev_good_rr = None
    for i in range(1, len(times)):
        rr = times[i] - times[i - 1]

        if not (RR_MIN_MS <= rr <= RR_MAX_MS):
            continue

        if prev_good_rr is not None:
            jump_frac = abs(rr - prev_good_rr) / max(prev_good_rr, 1e-6)
            if jump_frac > MAX_RR_JUMP_FRAC:
                continue

        rr_times.append(times[i])
        rr_vals.append(rr)
        prev_good_rr = rr

    return pd.DataFrame({
        "time_ms": rr_times,
        "rr_ms": rr_vals,
    })


def compute_freq_features(rr_win_ms: np.ndarray):
    """
    Estimate LF/HF from irregular RR intervals via simple interpolation.
    Returns (LF_power, HF_power, LFHF).
    If scipy is unavailable or data is insufficient, returns NaNs.
    """
    if (not SCIPY_OK) or len(rr_win_ms) < 8:
        return np.nan, np.nan, np.nan

    # Build cumulative beat times in seconds
    rr_s = rr_win_ms / 1000.0
    t_beats = np.cumsum(rr_s)
    t_beats = t_beats - t_beats[0]

    dur = t_beats[-1] - t_beats[0]
    if dur < 20.0:
        return np.nan, np.nan, np.nan

    # interpolate RR tachogram to uniform grid
    fs = 4.0
    t_uniform = np.arange(0.0, dur, 1.0 / fs)
    if len(t_uniform) < 16:
        return np.nan, np.nan, np.nan

    rr_uniform = np.interp(t_uniform, t_beats, rr_s)
    rr_uniform = rr_uniform - np.mean(rr_uniform)

    f, pxx = welch(rr_uniform, fs=fs, nperseg=min(256, len(rr_uniform)))

    lf_band = (f >= 0.04) & (f < 0.15)
    hf_band = (f >= 0.15) & (f < 0.40)

    if not np.any(lf_band) or not np.any(hf_band):
        return np.nan, np.nan, np.nan

    lf_power = float(np.trapz(pxx[lf_band], f[lf_band]))
    hf_power = float(np.trapz(pxx[hf_band], f[hf_band]))

    if hf_power <= 1e-12:
        lfhf = np.nan
    else:
        lfhf = float(lf_power / hf_power)

    return lf_power, hf_power, lfhf


def compute_window_features(
    raw_win: pd.DataFrame,
    rr_df: pd.DataFrame,
    win_start_ms: float,
    win_end_ms: float,
):
    rr_win = rr_df[(rr_df["time_ms"] >= win_start_ms) & (rr_df["time_ms"] < win_end_ms)].copy()
    rr = rr_win["rr_ms"].to_numpy(dtype=float)

    if len(rr) < MIN_RR_COUNT:
        return None

    diff_rr = np.diff(rr)
    hr = 60000.0 / rr

    # Beat density: beats per second in the window
    beat_count = len(rr)
    beat_density = beat_count / ((win_end_ms - win_start_ms) / 1000.0)

    # Motion from accelerometer magnitude std
    raw_seg = raw_win[(raw_win["time_ms"] >= win_start_ms) & (raw_win["time_ms"] < win_end_ms)].copy()
    motion_std = float(raw_seg["accmag"].std(ddof=1)) if len(raw_seg) >= 2 else np.nan

    lf_power, hf_power, lfhf = compute_freq_features(rr)

    feat = {
        "subject": SUBJECT_ID,
        "phase": PHASE_NAME,
        "label": LABEL_VALUE,
        "window_start": int(win_start_ms),

        "MeanNN": float(np.mean(rr)),
        "MedianNN": float(np.median(rr)),
        "IQRNN": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
        "MADNN": float(median_abs_deviation(rr)),
        "SDNN": float(np.std(rr, ddof=1)) if len(rr) >= 2 else np.nan,
        "RMSSD": float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) >= 1 else np.nan,

        "MeanHR": float(np.mean(hr)),
        "StdHR": float(np.std(hr, ddof=1)) if len(hr) >= 2 else np.nan,

        "LF_power": lf_power,
        "HF_power": hf_power,
        "LFHF": lfhf,

        "BeatDensity": float(beat_density),
        "motion_std": motion_std,
    }
    return feat


# ============================================================
# MAIN
# ============================================================
print(f"Reading raw stream: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)

required_raw_stream_cols = [
    "time_ms", "ir", "ppg", "beat", "bpm", "avg_bpm", "finger",
    "ax", "ay", "az", "accmag"
]
missing = [c for c in required_raw_stream_cols if c not in df.columns]
if missing:
    raise ValueError(f"Raw stream CSV missing columns: {missing}")

df = df.copy()
df = df.sort_values("time_ms").reset_index(drop=True)

# Basic cleanup
for c in required_raw_stream_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["time_ms", "beat", "finger", "accmag"]).copy()

print("Computing RR intervals from beat timestamps...")
rr_df = compute_rr_from_beats(df)
print(f"Accepted RR count: {len(rr_df)}")

if len(rr_df) == 0:
    raise RuntimeError("No usable RR intervals found. Check beat detection / finger contact.")

rr_df.to_csv("ppg_rr_intervals_clean.csv", index=False)
print("Saved clean RR: ppg_rr_intervals_clean.csv")

t0 = float(df["time_ms"].min())
t1 = float(df["time_ms"].max())

window_ms = WINDOW_SEC * 1000.0
step_ms = STEP_SEC * 1000.0

rows = []
win_start = t0
while win_start + window_ms <= t1:
    win_end = win_start + window_ms
    feat = compute_window_features(df, rr_df, win_start, win_end)
    if feat is not None:
        rows.append(feat)
    win_start += step_ms

feat_df = pd.DataFrame(rows)

if len(feat_df) == 0:
    raise RuntimeError("No valid windows created. Try longer recording or lower MIN_RR_COUNT.")

# Match train/inference raw required columns order
ordered_cols = [
    "subject", "phase", "label", "window_start",
    "MeanNN", "MedianNN", "IQRNN", "MADNN",
    "SDNN", "RMSSD",
    "MeanHR", "StdHR",
    "LF_power", "HF_power", "LFHF",
    "BeatDensity", "motion_std",
]
feat_df = feat_df[ordered_cols]

feat_df.to_csv(OUT_PATH, index=False)
print(f"Saved feature windows: {OUT_PATH}")
print(f"Rows: {len(feat_df)}")
print("\nHead:")
print(feat_df.head().to_string(index=False))