import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Charger les données nettoyées
df = pd.read_csv("data/processed/clean_weather.csv")

# Sélectionner les colonnes utiles
X = df[["humidity", "pressure"]]         # variables explicatives
y = df["temperature"]                    # variable à prédire

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Modèle entraîné avec MAE : {mae:.2f} | R² : {r2:.2f}")

# Sauvegarder le modèle
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/weather_model.pkl")
print("💾 Modèle sauvegardé dans : models/weather_model.pkl")
