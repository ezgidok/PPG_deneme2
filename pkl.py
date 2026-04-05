import pickle
from pathlib import Path

# Dosya yolunu oluşturma
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dataset" / "WESAD" / "S2" / "S2.pkl"

print("Dosya okunuyor, lütfen bekleyin...") # İşlemin başladığını anlamak için

try:
    with open(file_path, "rb") as f:
        # WESAD pkl dosyaları genellikle latin1 encoding gerektirir
        data = pickle.load(f, encoding='latin1')
    
    print("--- Dosya Başarıyla Okundu ---")
    
    # 1. Sözlüğün ana anahtarlarını gör (signal, label, subject)
    print("\nAna Başlıklar:", data.keys())
    
    # 2. Sinyal türlerini gör (chest, wrist)
    print("Cihazlar:", data['signal'].keys())
    
    # 3. Bilek (Wrist) cihazındaki sensörleri gör (BVP, EDA, TEMP, ACC)
    print("Bilek Sensörleri:", data['signal']['wrist'].keys())
    
    # 4. Senin için en önemli olan PPG (BVP) verisinin boyutuna bak
    ppg_data = data['signal']['wrist']['BVP']
    print(f"\nPPG (BVP) Veri Sayısı: {len(ppg_data)}")
    print("İlk 5 PPG değeri:", ppg_data[:5].flatten())

except FileNotFoundError:
    print(f"Hata: Dosya bulunamadı! Lütfen yolu kontrol et: {file_path}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")