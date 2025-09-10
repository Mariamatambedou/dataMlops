from data_prep import clean_weather_data

input_file = "data/raw/weatherHistory.csv"  # ou le nom réel du fichier dans data/raw/
output_file = "data/processed/clean_weather.csv"

clean_weather_data(input_file, output_file)
