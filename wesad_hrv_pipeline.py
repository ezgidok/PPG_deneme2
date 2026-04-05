from pathlib import Path
import numpy as np
import pandas as pd
import time

BASE_DIR = Path(__file__).resolve().parent
WESAD_DIR = BASE_DIR / "dataset" / "WESAD"


# ------------------------------------------------------------
# 1) QUEST OKU
# ------------------------------------------------------------
def parse_quest_times(quest_path: Path):
    order = start = end = None
    with open(quest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# ORDER"):
                order = [x.strip() for x in line.split(";")[1:] if x.strip()]
            elif line.startswith("# START"):
                start = [x.strip() for x in line.split(";")[1:] if x.strip()]
            elif line.startswith("# END"):
                end = [x.strip() for x in line.split(";")[1:] if x.strip()]

    if not (order and start and end):
        raise ValueError(f"ORDER/START/END satırları eksik: {quest_path}")

    phases = {}
    for name, s, e in zip(order, start, end):
        phases[name] = (float(s) * 60.0, float(e) * 60.0)  # dk -> sn
    return phases


# ------------------------------------------------------------
# 2) IBI OKU
# ------------------------------------------------------------
def load_ibi(ibi_path: Path):
    with open(ibi_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    data = np.loadtxt(ibi_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    t = data[:, 0]
    ibi = data[:, 1]
    return header, t, ibi


# ------------------------------------------------------------
# 3) ACC OKU
# ------------------------------------------------------------
def load_acc(acc_path: Path, retries: int = 3, sleep_sec: float = 0.5):
    """
    WESAD Empatica E4 ACC.csv için sağlam parser.
    Stale NFS / geçici dosya erişim sorunlarında birkaç kez tekrar dener.
    """
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            with open(acc_path, "r", encoding="utf-8") as f:
                line1 = f.readline().strip()
                line2 = f.readline().strip()

            def first_number(s: str) -> float:
                return float(s.split(",")[0].strip())

            start_ts = first_number(line1)
            fs = first_number(line2)

            data = np.loadtxt(acc_path, delimiter=",", skiprows=2)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            n = data.shape[0]
            t = np.arange(n, dtype=float) / fs
            mag = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))

            return start_ts, fs, t, mag

        except OSError as e:
            last_err = e
            print(f"[WARN] ACC open failed ({acc_path}) attempt {attempt}/{retries}: {e}")
            time.sleep(sleep_sec)

    raise last_err


# ------------------------------------------------------------
# 4) PENCERELEME
# ------------------------------------------------------------
def generate_windows(t_start, t_end, win_len=60.0, step=30.0):
    windows = []
    s = t_start
    while s + win_len <= t_end:
        windows.append((s, s + win_len))
        s += step
    return windows

def generate_windows_for_phase(phases, phase_name, win_len=60.0, step=30.0, start_buffer=30.0, end_buffer=30.0):
    if phase_name not in phases:
        return []

    raw_start, raw_end = phases[phase_name]
    t_start = raw_start + start_buffer
    t_end = raw_end - end_buffer

    if t_end - t_start < win_len:
        return []

    return generate_windows(t_start, t_end, win_len=win_len, step=step)


# ------------------------------------------------------------
# 5) LABEL
# ------------------------------------------------------------
def label_window(mid_t, phases):
    phase_name = None
    for name, (s, e) in phases.items():
        if s <= mid_t <= e:
            phase_name = name
            break

    if phase_name is None:
        return None, None

    if phase_name == "TSST":
        return 1, phase_name

    if phase_name in ["Base", "Medi 1", "Medi 2", "Fun"]:
        return 0, phase_name

    return None, phase_name


# ------------------------------------------------------------
# 6) HRV FEATURE
# ------------------------------------------------------------
def hrv_features(ibi_window, window_len_sec=60.0):
    ibi_window = np.asarray(ibi_window, dtype=float)

    # minimum beat sayısı
    if len(ibi_window) < 10:
        return None

    mean_nn = float(np.mean(ibi_window))
    median_nn = float(np.median(ibi_window))
    min_nn = float(np.min(ibi_window))
    max_nn = float(np.max(ibi_window))

    # robust variability
    iqr_nn = float(np.percentile(ibi_window, 75) - np.percentile(ibi_window, 25))
    mad_nn = float(np.median(np.abs(ibi_window - median_nn)))

    sdnn = float(np.std(ibi_window, ddof=1)) if len(ibi_window) > 1 else 0.0

    diffs = np.diff(ibi_window)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0

    # pNN50 (50 ms = 0.05 s)
    nn50 = np.sum(np.abs(diffs) > 0.05)
    pnn50 = float(nn50 / len(diffs)) if len(diffs) > 0 else 0.0

    # HR features
    hr_series = 60.0 / ibi_window
    mean_hr = float(np.mean(hr_series)) if len(hr_series) > 0 else 0.0
    std_hr = float(np.std(hr_series, ddof=1)) if len(hr_series) > 1 else 0.0

    # Poincaré SD1/SD2
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    sd1 = float(np.sqrt(0.5) * std_diff)
    sd2_term = (2.0 * (sdnn ** 2)) - (0.5 * (std_diff ** 2))
    sd2 = float(np.sqrt(sd2_term)) if sd2_term > 0 else 0.0

    # signal quality proxy
    ibi_outlier_ratio = float(np.mean((ibi_window < 0.3) | (ibi_window > 2.0)))
    beat_density = float(len(ibi_window) / window_len_sec)

    # frequency-domain approx
    lf_power = 0.0
    hf_power = 0.0
    lfhf = 0.0

    if len(ibi_window) >= 16:
        try:
            from scipy.signal import welch

            # uniformly sampled pseudo-series
            x_old = np.linspace(0.0, 1.0, len(ibi_window))
            x_new = np.linspace(0.0, 1.0, 4 * len(ibi_window))
            ibi_interp = np.interp(x_new, x_old, ibi_window)

            f, pxx = welch(ibi_interp, fs=4.0, nperseg=min(256, len(ibi_interp)))

            lf_mask = (f >= 0.04) & (f < 0.15)
            hf_mask = (f >= 0.15) & (f < 0.40)

            lf_power = float(np.trapz(pxx[lf_mask], f[lf_mask])) if np.any(lf_mask) else 0.0
            hf_power = float(np.trapz(pxx[hf_mask], f[hf_mask])) if np.any(hf_mask) else 0.0
            lfhf = float(lf_power / hf_power) if hf_power > 1e-12 else 0.0
        except Exception:
            pass

    return {
        "MeanNN": mean_nn,
        "MedianNN": median_nn,
        "MinNN": min_nn,
        "MaxNN": max_nn,
        "IQRNN": iqr_nn,
        "MADNN": mad_nn,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "MeanHR": mean_hr,
        "StdHR": std_hr,
        "SD1": sd1,
        "SD2": sd2,
        "LF_power": lf_power,
        "HF_power": hf_power,
        "LFHF": lfhf,
        "IBI_outlier_ratio": ibi_outlier_ratio,
        "BeatDensity": beat_density,
    }


# ------------------------------------------------------------
# 7) TEK SUBJECT İŞLE
# ------------------------------------------------------------
def process_subject(subj_dir: Path):
    subj = subj_dir.name

    quest_path = subj_dir / f"{subj}_quest.csv"
    if not quest_path.exists():
        print(f"[SKIP] {subj}: quest yok -> {quest_path}")
        return None

    ibi_candidates = [
        subj_dir / f"{subj}_E4_Data" / "IBI.csv",
        subj_dir / f"{subj}_E4_Data 2" / "IBI.csv",
    ]
    ibi_path = next((p for p in ibi_candidates if p.exists()), None)
    if ibi_path is None:
        print(f"[SKIP] {subj}: IBI yok")
        return None

    acc_candidates = [
        subj_dir / f"{subj}_E4_Data" / "ACC.csv",
        subj_dir / f"{subj}_E4_Data 2" / "ACC.csv",
    ]
    acc_path = next((p for p in acc_candidates if p.exists()), None)
    if acc_path is None:
        print(f"[SKIP] {subj}: ACC yok")
        return None

    try:
        _, _, t_acc, acc_mag = load_acc(acc_path)
    except Exception as e:
        print(f"[SKIP] {subj}: ACC okunamadı -> {e}")
        return None

    try:
        phases = parse_quest_times(quest_path)
        _, t_ibi, ibi = load_ibi(ibi_path)
    except Exception as e:
        print(f"[SKIP] {subj}: quest/IBI okunamadı -> {e}")
        return None

    main_phases = ["Base", "TSST", "Medi 1", "Fun", "Medi 2"]
    for p in main_phases:
        if p not in phases:
            print(f"[SKIP] {subj}: faz eksik -> {p}")
            return None

    # Window ayarları
    WIN_LEN = 90.0
    STEP = 30.0
    START_BUFFER = 30.0
    END_BUFFER = 30.0

    windows = []
    for phase_name in main_phases:
        phase_windows = generate_windows_for_phase(
            phases=phases,
            phase_name=phase_name,
            win_len=WIN_LEN,
            step=STEP,
            start_buffer=START_BUFFER,
            end_buffer=END_BUFFER,
        )
        windows.extend(phase_windows)

    rows = []
    for ws, we in windows:
        mid = (ws + we) / 2.0
        label, phase = label_window(mid, phases)
        if label is None:
            continue

        ibi_mask = (t_ibi >= ws) & (t_ibi < we)
        ibi_win = ibi[ibi_mask]

        acc_mask = (t_acc >= ws) & (t_acc < we)
        acc_win = acc_mag[acc_mask]
        motion_std = float(np.std(acc_win, ddof=1)) if len(acc_win) > 5 else 0.0

        feats = hrv_features(ibi_win, window_len_sec=(we - ws))
        if feats is None:
            continue

        row = {
            "subject": subj,
            "window_start": ws,
            "window_end": we,
            "phase": phase,
            "label": label,
            "n_beats": int(len(ibi_win)),
            "motion_std": motion_std,
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        print(f"[WARN] {subj}: hiç pencere üretilemedi")
        return None

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# 8) TÜM SUBJECTLERİ ÇALIŞTIR
# ------------------------------------------------------------
def main():
    if not WESAD_DIR.exists():
        raise FileNotFoundError(f"WESAD klasörü bulunamadı: {WESAD_DIR}")

    subject_dirs = sorted(
        [p for p in WESAD_DIR.iterdir() if p.is_dir() and p.name.startswith("S")]
    )
    print("Found subjects:", [p.name for p in subject_dirs])

    all_dfs = []
    for sd in subject_dirs:
        df = process_subject(sd)
        if df is not None:
            print(f"[OK] {sd.name}: rows={len(df)} label_counts={df['label'].value_counts().to_dict()}")
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Hiç subject işlenemedi. Dosya yapısını kontrol et.")

    all_df = pd.concat(all_dfs, ignore_index=True)

    out_path = BASE_DIR / "WESAD_HRV_windows_all.csv"
    all_df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("Total rows:", len(all_df))
    print("Subjects:", all_df["subject"].nunique())
    print("Label counts:", all_df["label"].value_counts().to_dict())
    print("Columns:", all_df.columns.tolist())


if __name__ == "__main__":
    main()