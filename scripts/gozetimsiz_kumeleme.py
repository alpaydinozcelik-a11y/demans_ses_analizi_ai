import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap

# 1. CSV dosyasını yükleyelim
# csv_olustur.py veriyi "../csv/ses_ozellikleri.csv" olarak kaydettiği için aynı yolu okuyoruz
df = pd.read_csv("../csv/ses_ozellikleri.csv")

# Sadece sayısal MFCC özelliklerini alalım (dosya_adi sütununu dışarıda bırakıyoruz)
X = df.drop(columns=["dosya_adi"]).values

# 2. Özellikleri Ölçeklendirme (K-Means'in doğru çalışması için şarttır)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. UMAP ile Boyut Azaltma
# 13 adet MFCC özelliğini hem daha iyi kümelemek hem de ekranda görebilmek için 2 boyuta düşürüyoruz
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_embedding = reducer.fit_transform(X_scaled)

# 4. K-Means ile Gözetimsiz Kümeleme
# Etiket vermiyoruz, sadece veriyi en mantıklı 2 gruba ayırmasını istiyoruz (Sağlıklı vs Hasta)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['tahmin_kumesi'] = kmeans.fit_predict(X_embedding)

# 5. Sonuçları csv klasörüne yeni bir rapor olarak kaydetme
df[["dosya_adi", "tahmin_kumesi"]].to_csv("../csv/kumeleme_sonuclari.csv", index=False)
print("Kümeleme tamamlandı! Sonuçlar 'csv/kumeleme_sonuclari.csv' dosyasına yazıldı.")

# 6. Grafik Olarak Ekrana Bastırma
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=df['tahmin_kumesi'], cmap='coolwarm', s=60)
plt.title("Gözetimsiz Ses Kümeleme Sonucu (Sağlıklı / Hasta Ayrımı)")
plt.xlabel("UMAP Boyut 1")
plt.ylabel("UMAP Boyut 2")
plt.colorbar(scatter, label="Atanan Küme ID (0 veya 1)")
plt.grid(True)
plt.show()
from sklearn.metrics import silhouette_score

# Ölçeklendirilmiş veri ve bulunan kümeler arasındaki silhouette skorunu hesapla
sil_score = silhouette_score(X_scaled, df['tahmin_kumesi'])
print(f"Modelin Gözetimsiz Kümeleme Kalitesi (Silhouette Skoru): {sil_score:.4f}")
# Her kümenin MFCC ortalamalarını hesaplayalım
kume_ortalamalari = df.groupby('tahmin_kumesi').mean(numeric_only=True)

# Grafiğe dökelim
plt.figure(figsize=(12, 6))
for kume in kume_ortalamalari.index:
    plt.plot(kume_ortalamalari.columns, kume_ortalamalari.loc[kume], marker='o', label=f'Küme {kume}')

plt.title("Kümelere Göre MFCC Profil Karşılaştırması (Karakteristik Farklar)")
plt.xlabel("Akustik Özellikler (MFCC)")
plt.ylabel("Ortalama Değerler")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
