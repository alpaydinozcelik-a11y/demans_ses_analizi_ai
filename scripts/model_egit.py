import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

# 1. Özellikleri ve oluşturduğumuz gözetimsiz kümeleri yükleyelim
df_features = pd.read_csv("../csv/ses_ozellikleri.csv")
df_labels = pd.read_csv("../csv/kumeleme_sonuclari.csv")

# Verileri birleştirelim
df = pd.merge(df_features, df_labels, on="dosya_adi")

X = df.drop(columns=["dosya_adi", "tahmin_kumesi"]).values
y = df["tahmin_kumesi"].values

# 2. Ölçekleyiciyi kur ve kaydet (Yeni sesler için de gerekecek)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 3. Canlı tahmin yapabilecek Derin/Sığ bir Yapay Sinir Ağı (MLP) eğitelim
# Ezberi önlemek için erken durdurma (early stopping) aktif
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42, early_stopping=True)
model.fit(X_scaled, y)

# 4. Eğitilen modeli gelecekte kullanmak üzere modeller klasörüne kaydet
import os
os.makedirs("../models", exist_ok=True)
with open("../models/tahmin_modeli.pkl", "wb") as f:
    pickle.dump(model, f)

print("Uygulama beyni (Model ve Scaler) başarıyla eğitildi ve 'models/' klasörüne kaydedildi!")