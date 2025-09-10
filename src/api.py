from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Charger le modèle sauvegardé
model = joblib.load("models/weather_model.pkl")

# Initialiser FastAPI
app = FastAPI()

# Schéma d'entrée avec Pydantic
class WeatherInput(BaseModel):
    humidity: float
    pressure: float

# Point d'entrée API
@app.post("/predict")
def predict(data: WeatherInput):
    input_df = pd.DataFrame([[data.humidity, data.pressure]], columns=["humidity", "pressure"])
    prediction = model.predict(input_df)[0]
    return {"predicted_temperature": round(prediction, 2)}
