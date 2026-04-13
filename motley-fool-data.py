from pathlib import Path

import pandas as pd

for name in ["motle-fool-data.pkl", "motley-fool-data.pkl"]:
    matches = list(Path(".").rglob(name))
    if matches:
        file_path = matches[0]
        break
else:
    raise FileNotFoundError("Couldn't find motle/motley-fool-data.pkl")

df = pd.read_pickle(file_path)
print(df.head())
