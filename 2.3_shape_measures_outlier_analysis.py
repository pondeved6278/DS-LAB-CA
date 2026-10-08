import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
root.destroy()
if not path:
    raise SystemExit

df = pd.read_csv(path)
df[["LAT", "LON"]] = df[["LAT", "LON"]].replace(0, np.nan)
cols = df.select_dtypes(include=np.number).columns

results = []
for c in cols:
    x = df[c].dropna()
    mean, median, std = x.mean(), x.median(), x.std()
    mode = x.mode()[0]
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    pearson_mode = (mean - mode) / std
    pearson_median = 3 * (mean - median) / std
    bowley = (q3 + q1 - 2 * median) / iqr if iqr != 0 else np.nan
    kurt = x.kurt()
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((x < lower) | (x > upper)).sum()
    results.append([c, pearson_mode, pearson_median, bowley, kurt,
                    q1, q3, iqr, lower, upper, n_out, 100 * n_out / len(x)])

res = pd.DataFrame(results, columns=[
    "Attribute", "Pearson_1(mode)", "Pearson_2(median)", "Bowley", "Kurtosis",
    "Q1", "Q3", "IQR", "Lower_Fence", "Upper_Fence", "Outliers", "Outlier_%"]).round(3)

print(res.to_string(index=False))
res.to_csv("summary.csv", index=False)

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, c in zip(axes.ravel(), cols):
    ax.boxplot(df[c].dropna(), vert=False)
    ax.set_title(c)
for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.show()
