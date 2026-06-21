import numpy as np
import librosa
import pickle
import sys

def canli_analiz_yap(ses_dosyasi_yolu):
    try:
        # 1. Modeli ve Ölçekleyiciyi yükle
        with open("../models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("../models/tahmin_modeli.pkl", "rb") as f:
            model = pickle.load(f)
            
        # 2. Dışarıdan gelen ses dosyasının MFCC özelliklerini çıkar
        ses, sr = librosa.load(ses_dosyasi_yolu, sr=None)
        mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)
        mfcc_ortalama = np.mean(mfcc, axis=1).reshape(1, -1)
        
        # 3. Veriyi ölçekle ve tahmin et
        mfcc_scaled = scaler.transform(mfcc_ortalama)
        tahmin = model.predict(mfcc_scaled)[0]
        olasilik = model.predict_proba(mfcc_scaled)[0]
        
        # 4. Sonucu ekrana yazdır (Önceki analizimize göre 0: Sağlıklı, 1: Hasta varsayımıyla)
        print("\n" + "="*40)
        print("          SES ANALİZ SONUÇLARI          ")
        print("="*40)
        if tahmin == 0:
            print(f"TEŞHİS: SAĞLIKLI BİREY (Güven Oranı: %{olasilik[0]*100:.2f})")
        else:
            print(f"TEŞHİS: PARKINSON / DEMANS RİSKİ (Güven Oranı: %{olasilik[1]*100:.2f})")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Analiz sırasında bir hata oluştu: {e}")

if __name__ == "__main__":
    import os
    import glob
    
    # 1. Kodun çalıştığı tam konumu otomatik bul
    mevcut_klasor = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Bir üst dizindeki data klasörünün tam (absolute) yolunu oluştur
    data_klasoru = os.path.join(mevcut_klasor, "..", "data")
    
    # 3. data klasörünün içindeki ve alt klasörlerindeki tüm .wav dosyalarını tara
    arama_kriteri = os.path.join(data_klasoru, "**", "*.wav")
    bulunan_sesler = glob.glob(arama_kriteri, recursive=True)
    
    if len(bulunan_sesler) > 0:
        # Bulduğu ilk .wav dosyasını test için seç
        test_sesi = bulunan_sesler[0] 
        print(f"\n[SİSTEM] Test edilecek ses dosyası otomatik bulundu:\n---> {test_sesi}")
        canli_analiz_yap(test_sesi)
    else:
        print(f"\n[HATA] {data_klasoru} dizininde hiç .wav dosyası bulunamadı. Lütfen klasörü kontrol edin.")