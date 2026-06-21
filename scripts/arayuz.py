import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import pickle
import os
import sounddevice as sd
from scipy.io.wavfile import write
import threading

# --- 1. ARKA PLAN İŞLEMLERİ (YAPAY ZEKA MOTORU & MİKROFON) ---

def dosya_sec():
    dosya_yolu = filedialog.askopenfilename(
        title="Analiz İçin Ses Dosyası Seçin",
        filetypes=[("Ses Dosyaları", "*.wav")]
    )
    if dosya_yolu:
        secili_dosya_etiketi.config(text=f"Seçilen: {os.path.basename(dosya_yolu)}")
        app.dosya_yolu = dosya_yolu
        sonuc_etiketi.config(text="Durum: Dosya yüklendi, analiz bekleniyor...", fg="black")

def canli_kayit_al():
    # Arayüzün donmaması için kaydı ayrı bir thread (iş parçacığı) olarak başlatıyoruz
    def kaydet():
        try:
            saniye = 5  # 5 saniyelik kayıt
            fs = 22050  # Librosa'nın standart çalışma frekansı
            
            btn_kayit.config(state="disabled", text="🔴 Kaydediliyor... Konuşun!")
            sonuc_etiketi.config(text="Lütfen konuşun (5 Saniye)...", fg="blue")
            
            # Mikrofonu aç ve dinle
            kayit = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
            sd.wait()  # Kaydın bitmesini bekle
            
            # Kaydı proje klasörüne geçici olarak kaydet
            kayit_yolu = "canli_kayit.wav"
            write(kayit_yolu, fs, kayit)
            
            app.dosya_yolu = kayit_yolu
            secili_dosya_etiketi.config(text="Seçilen: canli_kayit.wav (Mikrofon)")
            sonuc_etiketi.config(text="Durum: Kayıt tamamlandı, analize hazır!", fg="green")
            btn_kayit.config(state="normal", text="🎙️ Mikrofondan Sesimi Kaydet (5sn)")
            
        except Exception as e:
            messagebox.showerror("Mikrofon Hatası", f"Kayıt alınamadı:\n{e}")
            btn_kayit.config(state="normal", text="🎙️ Mikrofondan Sesimi Kaydet (5sn)")

    threading.Thread(target=kaydet).start()

def analiz_yap():
    if not hasattr(app, 'dosya_yolu') or not app.dosya_yolu:
        messagebox.showwarning("Uyarı", "Lütfen önce bir dosya seçin veya sesinizi kaydedin!")
        return
    
    try:
        # Modeli ve Ölçekleyiciyi yükle
        with open("../models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("../models/tahmin_modeli.pkl", "rb") as f:
            model = pickle.load(f)
            
        # Ses dosyasını işle
        ses, sr = librosa.load(app.dosya_yolu, sr=None)
        mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)
        mfcc_ortalama = np.mean(mfcc, axis=1).reshape(1, -1)
        
        # Tahmin et
        mfcc_scaled = scaler.transform(mfcc_ortalama)
        tahmin = model.predict(mfcc_scaled)[0]
        olasilik = model.predict_proba(mfcc_scaled)[0]
        
        # Sonucu arayüze yazdır
        if tahmin == 0:
            sonuc_metni = f"TEŞHİS: SAĞLIKLI BİREY\nGüven Oranı: %{olasilik[0]*100:.2f}"
            sonuc_etiketi.config(text=sonuc_metni, fg="#006400")
        else:
            sonuc_metni = f"TEŞHİS: PARKINSON / DEMANS RİSKİ\nGüven Oranı: %{olasilik[1]*100:.2f}"
            sonuc_etiketi.config(text=sonuc_metni, fg="#8B0000")
            
    except Exception as e:
        messagebox.showerror("Hata", f"Analiz sırasında bir hata oluştu:\n{e}")

# --- 2. GÖRSEL ARAYÜZ (GUI) TASARIMI ---

app = tk.Tk()
app.title("Demans ve Parkinson Analiz Sistemi | Eğitmen Alp Aydın Özçelik")
app.geometry("550x450")
app.configure(bg="#f4f4f9")

baslik = tk.Label(app, text="Yapay Zeka Destekli Canlı Ses Analizi", font=("Arial", 16, "bold"), bg="#f4f4f9", fg="#333333")
baslik.pack(pady=20)

secili_dosya_etiketi = tk.Label(app, text="Seçilen: Henüz ses girilmedi", font=("Arial", 11), bg="#f4f4f9", fg="#555555")
secili_dosya_etiketi.pack(pady=5)

# --- BUTONLAR ---
buton_cercevesi = tk.Frame(app, bg="#f4f4f9")
btn_kayit = tk.Button(buton_cercevesi, text="🎙️ Mikrofondan Sesimi Kaydet (5sn)", command=canli_kayit_al, width=28, height=2, font=("Arial", 11, "bold"), fg="darkred")
btn_kayit.grid(row=0, column=0, padx=5, pady=10)

btn_sec = tk.Button(buton_cercevesi, text="📂 Dosyadan Seç (.wav)", command=dosya_sec, width=28, height=2, font=("Arial", 11))
btn_sec.grid(row=1, column=0, padx=5, pady=5)
buton_cercevesi.pack()

btn_analiz = tk.Button(app, text="🔍 Yapay Zeka Analizini Başlat", command=analiz_yap, width=30, height=2, font=("Arial", 13, "bold"), bg="#4CAF50")
btn_analiz.pack(pady=20)

sonuc_etiketi = tk.Label(app, text="Durum: Bekleniyor...", font=("Arial", 14, "bold"), bg="#f4f4f9")
sonuc_etiketi.pack(pady=10)

app.mainloop()