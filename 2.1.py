import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
VALUE_COL = "Vict Age"
WEIGHT_COL = "Part 1-2"
N_BINS = 8


def load_data(path):
    df = pd.read_csv(path, usecols=[VALUE_COL, WEIGHT_COL], low_memory=False)
    df = df[df[VALUE_COL] > 0].dropna().reset_index(drop=True)
    df["weight"] = df[WEIGHT_COL].map({1: 2, 2: 1})
    return df


def ungrouped_measures(x, w):
    arithmetic = x.mean()
    weighted = np.average(x, weights=w)
    geometric = stats.gmean(x)
    median = np.median(x)
    mode_res = stats.mode(x, keepdims=False)
    return {
        "Arithmetic Mean": arithmetic,
        "Weighted Mean": weighted,
        "Geometric Mean": geometric,
        "Median": median,
        "Mode": mode_res.mode,
    }


def build_frequency_table(df, n_bins):
    edges = np.linspace(df[VALUE_COL].min(), df[VALUE_COL].max(), n_bins + 1)
    df = df.copy()
    df["class"] = pd.cut(df[VALUE_COL], bins=edges, include_lowest=True)
    table = df.groupby("class", observed=False).agg(
        f=(VALUE_COL, "size"),
        w=("weight", "sum"),
    ).reset_index()
    table["lower"] = edges[:-1]
    table["upper"] = edges[1:]
    table["mid"] = (table["lower"] + table["upper"]) / 2
    table["cf"] = table["f"].cumsum()
    return table, edges[1] - edges[0]


def grouped_measures(table, h):
    n = table["f"].sum()
    f = table["f"].values
    mid = table["mid"].values

    arithmetic = (f * mid).sum() / n

    weighted = (table["w"].values * mid).sum() / table["w"].sum()

    geometric = np.exp((f * np.log(mid)).sum() / n)

    median_idx = np.argmax(table["cf"].values >= n / 2)
    L = table.loc[median_idx, "lower"]
    cf_prev = table.loc[median_idx - 1, "cf"] if median_idx > 0 else 0
    f_med = table.loc[median_idx, "f"]
    median = L + ((n / 2 - cf_prev) / f_med) * h

    mode_idx = np.argmax(f)
    L_m = table.loc[mode_idx, "lower"]
    f1 = f[mode_idx]
    f0 = f[mode_idx - 1] if mode_idx > 0 else 0
    f2 = f[mode_idx + 1] if mode_idx < len(f) - 1 else 0
    denom = 2 * f1 - f0 - f2
    mode = L_m + ((f1 - f0) / denom) * h if denom != 0 else table.loc[mode_idx, "mid"]

    return {
        "Arithmetic Mean": arithmetic,
        "Weighted Mean": weighted,
        "Geometric Mean": geometric,
        "Median": median,
        "Mode": mode,
    }


def main():
    df = load_data(CSV_PATH)
    x = df[VALUE_COL].values
    w = df["weight"].values

    print(f"Attribute: {VALUE_COL} | Records: {len(df):,}")
    print(f"Weights: {WEIGHT_COL} (Part 1 = 2, Part 2 = 1)\n")

    ungrouped = ungrouped_measures(x, w)

    table, h = build_frequency_table(df, N_BINS)
    print("Grouped frequency table:")
    print(table[["class", "f", "w", "mid", "cf"]].to_string(index=False))
    print(f"\nClass width: {h:.4f}\n")

    grouped = grouped_measures(table, h)

    comparison = pd.DataFrame({
        "Ungrouped": ungrouped,
        "Grouped": grouped,
    })
    comparison["Difference"] = comparison["Ungrouped"] - comparison["Grouped"]
    print(comparison.round(4).to_string())

    out_path = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\central_tendency.csv"
    comparison.round(4).to_csv(out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()