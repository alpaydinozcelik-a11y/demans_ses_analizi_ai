import os
import librosa
import numpy as np
import pandas as pd

veriler = []
veri_klasoru = "../data" # Sol taraftaki dosya yapına göre 'data' klasörün bir üst dizinde

for klasor in os.listdir(veri_klasoru):
    if klasor.startswith("Process-rec"):
        klasor_yolu = os.path.join(veri_klasoru, klasor)
        
        for dosya in os.listdir(klasor_yolu):
            if dosya.endswith(".wav"):
                dosya_yolu = os.path.join(klasor_yolu, dosya)
                
                # Sesi yükle (sr=None orijinal örnekleme hızını korur)
                ses, sr = librosa.load(dosya_yolu, sr=None)
                
                # 13 adet MFCC çıkarıyoruz (Kodunda 13 seçmişsin, gayet iyi)
                mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)
                
                # Zaman boyutunun ortalamasını alıp 13 boyutlu bir vektör elde ediyoruz
                mfcc_ortalama = np.mean(mfcc, axis=1)
                
                # Dosya adını ve çıkarılan 13 özelliği bir sözlükte toplayalım
                satir = {"dosya_adi": dosya}
                for i in range(13):
                    satir[f"mfcc_{i+1}"] = mfcc_ortalama[i]
                
                veriler.append(satir)

# Verileri Pandas DataFrame'e çevirip CSV olarak kaydedelim
df = pd.DataFrame(veriler)

# Sol taraftaki yapıda 'csv' klasörün var, oraya kaydediyoruz
os.makedirs("../csv", exist_ok=True)
df.to_csv("../csv/ses_ozellikleri.csv", index=False)
print("Özellik çıkarma tamamlandı ve csv/ses_ozellikleri.csv dosyasına kaydedildi!")