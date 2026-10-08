import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SKIP = {"DR_NO", "AREA", "Rpt Dist No", "Part 1-2", "Crm Cd", "Premis Cd", "Weapon Used Cd", "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"}

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
root.destroy()
if not path:
    raise SystemExit

df = pd.read_csv(path)
X = df[[c for c in df.select_dtypes(np.number) if c not in SKIP]]

results = {
    "Original": X,
    "Min-Max": (X - X.min()) / (X.max() - X.min()),
    "Z-score": (X - X.mean()) / X.std(ddof=0),
    "Decimal Scaling": X / 10 ** (np.floor(np.log10(X.abs().max())) + 1),
}

pd.set_option("display.width", 200)
print(pd.concat({k: v.describe().loc[["min", "max", "mean", "std"]] for k, v in results.items()}).round(4))

fig, axes = plt.subplots(len(results), X.shape[1], figsize=(4 * X.shape[1], 3 * len(results)))
for i, (name, data) in enumerate(results.items()):
    for j, col in enumerate(X.columns):
        axes[i, j].hist(data[col], bins=50)
        axes[i, j].set_title(f"{name}: {col}", fontsize=9)
plt.tight_layout()
plt.show()
