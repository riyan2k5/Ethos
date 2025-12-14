import pandas as pd

df = pd.read_csv("spotify_data_mapped_genres.csv", nrows=5)
print("Columns:", list(df.columns))
