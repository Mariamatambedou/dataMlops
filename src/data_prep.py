import pandas as pd
import os

def clean_weather_data(input_path, output_path):
    print(f"📥 Lecture de : {input_path}")
    df = pd.read_csv(input_path)

    print("📊 Nombre de lignes avant nettoyage :", len(df))
    print("📄 Colonnes :", df.columns.tolist())

    # Conversion de la date
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce')

    # Supprimer les lignes avec des valeurs manquantes
    df.dropna(inplace=True)

    # Renommer les colonnes pour simplifier
    df.rename(columns={
        'Formatted Date': 'date',
        'Temperature (C)': 'temperature',
        'Humidity': 'humidity',
        'Pressure (millibars)': 'pressure'
    }, inplace=True)

    # Garder uniquement les colonnes utiles
    selected_cols = ['date', 'temperature', 'humidity', 'pressure']
    df = df[selected_cols]

    # Créer le dossier si besoin
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Enregistrement
    df.to_csv(output_path, index=False)
    print(f"✅ {len(df)} lignes nettoyées enregistrées dans : {output_path}")
