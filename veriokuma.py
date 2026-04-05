from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# A) DOSYA YOLLARI (S2 için sabit)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

SUBJ = "S2"
SUBJ_DIR = BASE_DIR / "dataset" / "WESAD" / SUBJ

quest_path = SUBJ_DIR / f"{SUBJ}_quest.csv"
ibi_path   = SUBJ_DIR / f"{SUBJ}_E4_Data" / "IBI.csv"  # doğru klasör

print("quest_path:", quest_path, "exists?", quest_path.exists())
print("ibi_path:", ibi_path, "exists?", ibi_path.exists())

if not quest_path.exists():
    raise FileNotFoundError(f"Quest dosyası bulunamadı: {quest_path}")
if not ibi_path.exists():
    raise FileNotFoundError(f"IBI dosyası bulunamadı: {ibi_path}")


# ------------------------------------------------------------
# B) QUEST OKU: Faz başlangıç/bitişlerini dakika->saniye çevir
# ------------------------------------------------------------
def parse_quest_times(quest_path):
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
        raise ValueError("ORDER/START/END satırları bulunamadı veya eksik.")

    phases = {}
    for name, s, e in zip(order, start, end):
        phases[name] = (float(s) * 60.0, float(e) * 60.0)  # dk -> sn

    return phases


# ------------------------------------------------------------
# C) IBI OKU: (t_offset_sec, ibi_sec)
# ------------------------------------------------------------
def load_ibi(ibi_path):
    with open(ibi_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    data = np.loadtxt(ibi_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    ibi = data[:, 1]
    return header, t, ibi


phases = parse_quest_times(quest_path)
header, t_ibi, ibi = load_ibi(ibi_path)

print("\nOK - IBI header:", header)
print("Phase keys:", list(phases.keys()))
print("IBI count:", len(ibi))
print("t range:", float(t_ibi.min()), "->", float(t_ibi.max()))
print("ibi range:", float(ibi.min()), "->", float(ibi.max()))


# ------------------------------------------------------------
# D) (KONTROL) Fazları saniye cinsinden yazdır
# ------------------------------------------------------------
print("\n--- Phase times (sec) ---")
for name, (s, e) in phases.items():
    print(f"{name:>6}: {s:8.1f} -> {e:8.1f}  (dur={(e-s):.1f}s)")


# ------------------------------------------------------------
# E) PENCERE ÜRET: 60 sn, %50 overlap (30 sn kaydır)
# ------------------------------------------------------------
def generate_windows(t_start, t_end, win_len=60.0, step=30.0):
    windows = []
    s = t_start
    while s + win_len <= t_end:
        windows.append((s, s + win_len))
        s += step
    return windows


# sadece bu fazları kullanacağız (okuma fazlarını şimdilik dışarıda bırakıyoruz)
main_phases = ["Base", "TSST", "Medi 1", "Fun", "Medi 2"]

global_start = min(phases[p][0] for p in main_phases)
global_end   = max(phases[p][1] for p in main_phases)

windows = generate_windows(global_start, global_end, win_len=60.0, step=30.0)
print("\nWindow count:", len(windows))
print("First 3 windows:", windows[:3])
print("Last window:", windows[-1])


# ------------------------------------------------------------
# F) LABEL: pencere orta noktası hangi fazdaysa o etiket
#    TSST -> 1 (stress)
#    Base/Medi/Fun -> 0 (non-stress)
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

    return None, phase_name  # sRead/fRead vs.


# ------------------------------------------------------------
# G) HRV FEATURE: MeanNN, SDNN, RMSSD
# ------------------------------------------------------------
def hrv_features(ibi_window):
    # HRV için yeterli beat yoksa pencereyi at
    if len(ibi_window) < 5:
        return None

    mean_nn = float(np.mean(ibi_window))
    sdnn = float(np.std(ibi_window, ddof=1)) if len(ibi_window) > 1 else 0.0

    diffs = np.diff(ibi_window)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else 0.0

    return mean_nn, sdnn, rmssd


# ------------------------------------------------------------
# H) TÜM PENCERELER: HRV + label tabloya yaz
# ------------------------------------------------------------
rows = []
for (ws, we) in windows:
    mid = (ws + we) / 2.0

    label, phase = label_window(mid, phases)
    if label is None:
        continue

    mask = (t_ibi >= ws) & (t_ibi < we)
    ibi_win = ibi[mask]

    feats = hrv_features(ibi_win)
    if feats is None:
        continue

    mean_nn, sdnn, rmssd = feats

    rows.append({
        "window_start": ws,
        "window_end": we,
        "phase": phase,
        "label": label,
        "n_beats": int(len(ibi_win)),
        "MeanNN": mean_nn,
        "SDNN": sdnn,
        "RMSSD": rmssd,
    })

df = pd.DataFrame(rows)

print("\n--- HRV table preview ---")
print(df.head(10))

print("\nPhase counts:\n", df["phase"].value_counts())
print("\nLabel counts:\n", df["label"].value_counts())


# ------------------------------------------------------------
# I) HIZLI KARŞILAŞTIRMA (stres vs non-stres)
# ------------------------------------------------------------
print("\n--- Mean features by label ---")
print(df.groupby("label")[["MeanNN", "SDNN", "RMSSD"]].mean())


# ------------------------------------------------------------
# J) CSV OLARAK KAYDET (çıktı datasetin)
# ------------------------------------------------------------
out_path = BASE_DIR / "S2_HRV_windows.csv"
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)
