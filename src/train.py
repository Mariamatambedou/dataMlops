import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# On n'utilise MLflow que si on n'est pas en CI/CD
USE_MLFLOW = not os.environ.get("CI")  # GitHub Actions définit CI=true

if USE_MLFLOW:
    import mlflow
    import mlflow.sklearn

    # Définir l'expérience AVANT de démarrer la run
    mlflow.set_tracking_uri("./mlruns")  # chemin relatif compatible Windows
    mlflow.set_experiment("weather-forecast")

# Charger les données
df = pd.read_csv("data/processed/clean_weather.csv")
X = df[["humidity", "pressure"]]
y = df["temperature"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE : {mae:.2f} | R² : {r2:.2f}")

# MLflow : log métriques, modèle et tags
if USE_MLFLOW:
    with mlflow.start_run() as run:
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(
            model,
            "weather_model",
            input_example=X_test.iloc[:5]  # utile pour la production
        )
        mlflow.set_tag("team", "mariama")
        mlflow.set_tag("model_type", "LinearRegression")
        mlflow.set_tag("run_note", "Run locale pour test MLflow")

# Sauvegarder le modèle localement
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/weather_model.pkl")
print("💾 Modèle sauvegardé dans : models/weather_model.pkl")
