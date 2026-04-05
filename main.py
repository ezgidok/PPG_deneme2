import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("PPG_Dataset.xlsx", engine="openpyxl")

print("Ham shape:", df.shape)
print(df.head())


first_row = df.iloc[0].values
indices = np.arange(df.shape[1])


if np.allclose(first_row, indices, atol=1e-6):
    print("İlk satır index gibi görünüyor")
    df = df.iloc[1:].reset_index(drop=True)

print("Temiz shape:", df.shape)

#(num_samples, 1000, 1) 

X = df.values.astype("float32")
X = X[..., np.newaxis]  


print("Model input shape:", X.shape)

#veri çok büyükse sample al
#model için boyut azaltma değil
values = X.reshape(-1)
max_points = 500_000

if values.size > max_points:
    idx = np.random.choice(values.size, max_points, replace=False)
    values = values[idx]

plt.figure()
plt.hist(values, bins=100)
plt.title("PPG Histogram")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.savefig("ppg_histogram.png", dpi=150, bbox_inches="tight")


example = X[0].flatten()

plt.figure()
plt.plot(example)
plt.title("PPG Sinyali")
plt.xlabel("Index")
plt.ylabel("Genlik")
plt.savefig("ppg_sinyal.png", dpi=150, bbox_inches="tight")



print("Min:", float(values.min()))
print("Max:", float(values.max()))
print("Yüzdelikler:", np.percentile(values, [0, 1, 50, 99, 100]))

