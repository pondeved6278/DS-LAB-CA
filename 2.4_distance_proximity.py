import tkinter as tk
from tkinter import filedialog
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
root.destroy()
if not path:
    raise SystemExit

df = pd.read_csv(path)

X = df[["TIME OCC", "Vict Age", "LAT", "LON"]]
X = X.fillna(X.median())

B = df[["Weapon Desc", "Crm Cd 2", "Cross Street", "Mocodes"]].notna().assign(
    male=df["Vict Sex"].eq("M"), part1=df["Part 1-2"].eq(1)).astype(int)

T = (df["Crm Cd Desc"].fillna("") + " " + df["LOCATION"].fillna("")).str.split().str.join(" ")

idx = df.sample(6, random_state=42).index
labels = df.loc[idx, "DR_NO"].astype(str).tolist()
x, b, t = X.loc[idx].values, B.loc[idx].values.astype(bool), T.loc[idx].tolist()
z = ((X - X.mean()) / X.std(ddof=0)).loc[idx].values
VI = np.linalg.pinv(np.cov(X.values.T))


def minkowski(a, p):
    return (np.abs(a[:, None] - a[None]) ** p).sum(-1) ** (1 / p)


def edit(a, b):
    p = list(range(len(b) + 1))
    for i, u in enumerate(a, 1):
        c = [i]
        for j, v in enumerate(b, 1):
            c.append(min(p[j] + 1, c[-1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]


d = x[:, None] - x[None]
inter, union = (b[:, None] & b[None]).sum(-1), (b[:, None] | b[None]).sum(-1)
V = pd.DataFrame([Counter(s.split()) for s in t]).fillna(0).values
V = V / np.linalg.norm(V, axis=1, keepdims=True)

M = {
    "Minkowski (p=3)": minkowski(z, 3),
    "Euclidean": minkowski(z, 2),
    "Manhattan": minkowski(z, 1),
    "Mahalanobis": np.sqrt(np.einsum("ijk,kl,ijl->ij", d, VI, d)),
    "SMC distance": (b[:, None] != b[None]).mean(-1),
    "Jaccard distance": np.where(union == 0, 0, 1 - inter / np.maximum(union, 1)),
    "Cosine distance": np.clip(1 - V @ V.T, 0, None),
    "Edit distance": np.array([[edit(u, v) for v in t] for u in t]),
}

pd.set_option("display.width", 200)
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, (name, m) in zip(axes.ravel(), M.items()):
    print(f"\n{name}\n{pd.DataFrame(m, labels, labels).round(4)}")
    fig.colorbar(ax.imshow(m), ax=ax, fraction=0.046)
    ax.set_title(name)
    ax.set_xticks(range(6), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(6), labels, fontsize=7)
plt.tight_layout()
plt.show()
