import os
import zipfile

# Nom du dataset Kaggle
dataset = "muthuj7/weather-dataset"

# Dossier de destination
download_dir = "data/raw"

# Création du dossier
os.makedirs(download_dir, exist_ok=True)

# Téléchargement
os.system(f"kaggle datasets download -d {dataset} -p {download_dir} --unzip")

print(f"Données téléchargées et extraites dans {download_dir}")
