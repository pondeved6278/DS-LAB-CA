import pandas as pd
import numpy as np

df = pd.read_csv("export.csv", low_memory=False)

print("Missing values before handling:")
print(df.isnull().sum())

for column in ["Vict Sex", "Vict Descent", "Mocodes", "Weapon Desc", "Cross Street", "Premis Desc"]:
    df[column] = df[column].fillna("Unknown / Not Recorded")

df["Weapon Used Cd"] = df["Weapon Used Cd"].fillna(-1)

df.loc[
    (df["Vict Age"] <= 0) | (df["Vict Age"] > 100),
    "Vict Age"
] = np.nan

age_median = df["Vict Age"].median()
df["Vict Age"] = df["Vict Age"].fillna(age_median)

df.loc[
    (df["LAT"] == 0) & (df["LON"] == 0),
    ["LAT", "LON"]
] = np.nan

print("\nMissing values after handling:")
print(df.isnull().sum())

print("\nMedian used for Vict Age:", age_median)
