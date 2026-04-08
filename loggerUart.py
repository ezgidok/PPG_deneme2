import serial
import time
import pandas as pd
import numpy as np

PORT = "/dev/cu.usbmodem1101"   # <-- change this
BAUD = 115200

OUT_RAW = "ppg_stream_raw2.csv"
OUT_RR  = "ppg_rr_intervals2.csv"

print("Opening serial...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

columns = [
    "time_ms", "ir", "ppg", "beat", "bpm", "avg_bpm", "finger",
    "ax", "ay", "az", "accmag"
]

rows = []

print("Logging... Press CTRL+C to stop.")
try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != len(columns):
            continue

        try:
            row = [float(p) for p in parts]
            rows.append(row)
        except:
            continue

except KeyboardInterrupt:
    print("\nStopped.")

ser.close()

df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUT_RAW, index=False)
print(f"Saved raw: {OUT_RAW}")

# ---------------------------------------------------
# RR extraction
# ---------------------------------------------------

# Only valid beats with finger present
df_valid = df[(df["beat"] == 1) & (df["finger"] == 1)].copy()

times = df_valid["time_ms"].values

rr_list = []
rr_time = []

for i in range(1, len(times)):
    rr = times[i] - times[i-1]

    # basic physiological filtering
    if 300 < rr < 2000:   # 30–200 BPM
        rr_list.append(rr)
        rr_time.append(times[i])

rr_df = pd.DataFrame({
    "time_ms": rr_time,
    "rr_ms": rr_list
})

rr_df.to_csv(OUT_RR, index=False)
print(f"Saved RR: {OUT_RR}")

# Quick sanity stats
if len(rr_list) > 0:
    print("RR stats:")
    print(f"Mean RR: {np.mean(rr_list):.1f} ms")
    print(f"Std RR : {np.std(rr_list):.1f} ms")
    print(f"BPM    : {60000/np.mean(rr_list):.1f}")