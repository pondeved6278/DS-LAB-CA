import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
OUT_DIR = r"C:\Users\VED PONDE\OneDrive\Desktop\DS"
N_BINS = 5

COLUMNS = ["LAT", "LON", "Vict Age", "TIME OCC"]


def load_clean(path):
    df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    df = df[(df["LAT"] != 0) & (df["LON"] != 0)]
    df = df[df["Vict Age"] > 0]
    return df.dropna().reset_index(drop=True)


def equal_width(series, n_bins):
    return pd.cut(series, bins=n_bins, include_lowest=True, duplicates="drop")


def equal_frequency(series, n_bins):
    return pd.qcut(series, q=n_bins, duplicates="drop")


def histogram_bins(series, rule="auto"):
    counts, edges = np.histogram(series, bins=rule)
    return counts, edges


def summarize_binning(binned):
    summary = binned.value_counts().sort_index().rename("Count").to_frame()
    summary["Percent"] = (summary["Count"] / summary["Count"].sum() * 100).round(2)
    return summary


def plot_comparison(series, col, n_bins, out_dir):
    ew_counts, ew_edges = np.histogram(series, bins=n_bins)
    ef_edges = np.unique(series.quantile(np.linspace(0, 1, n_bins + 1)).values)
    ef_counts, _ = np.histogram(series, bins=ef_edges)
    h_counts, h_edges = histogram_bins(series, "auto")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].bar(ew_edges[:-1], ew_counts, width=np.diff(ew_edges), align="edge", edgecolor="black")
    axes[0].set_title(f"{col} - Equal-width ({n_bins} bins)")

    axes[1].bar(ef_edges[:-1], ef_counts, width=np.diff(ef_edges), align="edge", edgecolor="black")
    axes[1].set_title(f"{col} - Equal-frequency ({len(ef_counts)} bins)")

    axes[2].bar(h_edges[:-1], h_counts, width=np.diff(h_edges), align="edge", edgecolor="black")
    axes[2].set_title(f"{col} - Histogram analysis ({len(h_counts)} bins)")

    for ax in axes:
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.tight_layout()
    safe = col.replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"binning_{safe}.png"), dpi=150)
    plt.close(fig)


def main():
    df = load_clean(CSV_PATH)
    print(f"Rows after cleaning: {len(df):,}\n")

    result = pd.DataFrame(index=df.index)

    for col in COLUMNS:
        s = df[col]
        print("=" * 70)
        print(f"{col}  (min={s.min()}, max={s.max()}, mean={s.mean():.4f}, std={s.std():.4f})")
        print("=" * 70)

        ew = equal_width(s, N_BINS)
        ef = equal_frequency(s, N_BINS)
        counts, edges = histogram_bins(s, "auto")

        print(f"\nEqual-width binning ({N_BINS} bins):")
        print(summarize_binning(ew).to_string())

        print(f"\nEqual-frequency binning ({N_BINS} bins):")
        print(summarize_binning(ef).to_string())

        print(f"\nHistogram analysis (auto rule, {len(counts)} bins):")
        hist_table = pd.DataFrame({
            "Lower": edges[:-1].round(4),
            "Upper": edges[1:].round(4),
            "Count": counts,
            "Percent": (counts / counts.sum() * 100).round(2),
        })
        print(hist_table.to_string(index=False))

        peak = hist_table.loc[hist_table["Count"].idxmax()]
        print(f"\nPeak bin: [{peak['Lower']}, {peak['Upper']}) with {int(peak['Count']):,} records")
        print(f"Skewness: {s.skew():.4f} | Kurtosis: {s.kurt():.4f}\n")

        result[f"{col}_equal_width"] = ew.astype(str)
        result[f"{col}_equal_freq"] = ef.astype(str)
        result[f"{col}_hist_bin"] = pd.cut(s, bins=edges, include_lowest=True).astype(str)

        plot_comparison(s, col, N_BINS, OUT_DIR)

    out_path = os.path.join(OUT_DIR, "discretized_features.csv")
    result.to_csv(out_path, index=False)
    print(f"Saved discretized data -> {out_path}")
    print(f"Saved plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()