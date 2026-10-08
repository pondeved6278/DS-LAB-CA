import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("export.csv")

df[["LAT", "LON"]] = df[["LAT", "LON"]].replace(0, np.nan)
df["Vict Age"] = df["Vict Age"].where(df["Vict Age"] > 0)

cols = ["TIME OCC", "AREA", "Part 1-2", "Crm Cd", "Vict Age",
        "Premis Cd", "LAT", "LON"]
data = df[cols].dropna()

sample = data.sample(5000, random_state=42)

corr = data.corr()
print(corr.round(2))
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

pairs = [("LON", "LAT"), ("Vict Age", "TIME OCC"),
         ("Crm Cd", "Premis Cd"), ("AREA", "LAT")]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (xc, yc) in zip(axes.ravel(), pairs):
    ax.scatter(sample[xc], sample[yc], s=5, alpha=0.4)
    r = data[xc].corr(data[yc])
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    ax.set_title(f"{xc} vs {yc} (r = {r:.2f})")
plt.tight_layout()
plt.savefig("scatter_plots.png")
plt.show()

sns.pairplot(sample, corner=True, plot_kws={"s": 5, "alpha": 0.4})
plt.savefig("pairplot.png")
plt.show()

