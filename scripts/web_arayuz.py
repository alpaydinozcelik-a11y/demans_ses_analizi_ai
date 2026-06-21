import streamlit as st
import numpy as np
import librosa
import pickle
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Ses Analiz Sistemi", page_icon="🧠", layout="centered")

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🎙️ Yapay Zeka Destekli Ses Analizi")
st.subheader("Parkinson ve Demans Erken Teşhis Prototipi")
st.markdown("**Geliştirici:** Eğitmen Alp Aydın Özçelik (Yapay Zeka Mühendisliği)")
st.divider()

# --- MODELLERİ YÜKLEME (Önbelleğe alarak siteyi hızlandırıyoruz) ---
@st.cache_resource
def modeli_yukle():
    with open("/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("/models/tahmin_modeli.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

try:
    scaler, model = modeli_yukle()
except Exception as e:
    st.error(f"Yapay zeka modelleri yüklenemedi. Lütfen dosya yollarını kontrol edin: {e}")
    st.stop()

# --- KULLANICI ARAYÜZÜ (Dosya Yükleme veya Mikrofon) ---
st.write("### Analiz için bir ses verisi sağlayın:")

# Streamlit'in yerleşik dosya yükleyicisi ve mikrofon aracı
ses_dosyasi = st.file_uploader("📂 Bilgisayardan/Telefondan .wav dosyası seçin", type=["wav"])
st.write("veya")
ses_kaydi = st.audio_input("🎙️ Doğrudan mikrofondan sesinizi kaydedin")

# Hangisi doluysa onu işleme alacağız
islem_yapilacak_ses = ses_dosyasi if ses_dosyasi else ses_kaydi

if islem_yapilacak_ses is not None:
    # Yüklenen veya kaydedilen sesi dinletme oynatıcısı
    st.audio(islem_yapilacak_ses, format="audio/wav")
    
    if st.button("🔍 Yapay Zeka Analizini Başlat", use_container_width=True):
        with st.spinner('Sinyaller işleniyor ve yapay zeka tarafından analiz ediliyor...'):
            try:
                # Sesi librosa'nın okuyabilmesi için geçici olarak diske kaydet
                gecici_dosya = "gecici_ses.wav"
                with open(gecici_dosya, "wb") as f:
                    f.write(islem_yapilacak_ses.read())
                
                # 1. Özellik Çıkarımı (MFCC)
                ses, sr = librosa.load(gecici_dosya, sr=None)
                mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)
                mfcc_ortalama = np.mean(mfcc, axis=1).reshape(1, -1)
                
                # 2. Tahmin ve Olasılık Hesaplama
                mfcc_scaled = scaler.transform(mfcc_ortalama)
                tahmin = model.predict(mfcc_scaled)[0]
                olasilik = model.predict_proba(mfcc_scaled)[0]
                
                # 3. Sonuç Gösterimi
                st.divider()
                st.write("### 📊 Analiz Sonucu")
                
                if tahmin == 0:
                    st.success(f"**TEŞHİS:** SAĞLIKLI BİREY\n\n**Model Güven Oranı:** %{olasilik[0]*100:.2f}")
                else:
                    st.error(f"**TEŞHİS:** PARKINSON / DEMANS RİSKİ SİNYALİ\n\n**Model Güven Oranı:** %{olasilik[1]*100:.2f}")
                    
                # Çöp veriyi temizle
                os.remove(gecici_dosya)
                
            except Exception as e:
                st.error(f"İşlem sırasında teknik bir hata oluştu: {e}")
