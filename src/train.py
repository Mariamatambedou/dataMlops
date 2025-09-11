import mlflow
import mlflow.sklearn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ⚡ Ajouter cette ligne pour forcer MLflow à utiliser un stockage local
mlflow.set_tracking_uri("file:./mlruns")  # chemin relatif dans ton projet

# Charger les données
df = pd.read_csv("data/processed/clean_weather.csv")
X = df[["humidity", "pressure"]]
y = df["temperature"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Démarrer un run MLflow
with mlflow.start_run():
    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Évaluer
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ MAE : {mae:.2f} | R² : {r2:.2f}")

    # Enregistrer les métriques dans MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # Enregistrer le modèle dans MLflow
    mlflow.sklearn.log_model(model, "weather_model")

    # Optionnel : enregistrer aussi le joblib classique
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/weather_model.pkl")
    print("💾 Modèle sauvegardé dans : models/weather_model.pkl")
