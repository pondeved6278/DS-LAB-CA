import os
import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
OUT_DIR = r"C:\Users\VED PONDE\OneDrive\Desktop\DS"

COLUMNS = ["Vict Age", "TIME OCC", "LAT", "LON"]


def load_data(path):
    df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    df = df[(df["Vict Age"] > 0) & (df["LAT"] != 0) & (df["LON"] != 0)]
    return df.dropna().reset_index(drop=True)


def data_range(x):
    return x.max() - x.min()


def variance(x, sample=True):
    return np.var(x, ddof=1 if sample else 0)


def std_dev(x, sample=True):
    return np.std(x, ddof=1 if sample else 0)


def rmsd(x, reference=None):
    ref = x.mean() if reference is None else reference
    return np.sqrt(np.mean((x - ref) ** 2))


def coefficient_of_dispersion(x):
    q1, q3 = np.percentile(x, [25, 75])
    quartile_cd = (q3 - q1) / (q3 + q1)
    cv = std_dev(x) / x.mean() * 100 if x.mean() != 0 else np.nan
    mean_abs_dev_cd = np.mean(np.abs(x - np.median(x))) / np.median(x) if np.median(x) != 0 else np.nan
    return quartile_cd, cv, mean_abs_dev_cd


def compute_all(x):
    q_cd, cv, mad_cd = coefficient_of_dispersion(x)
    return {
        "Count": len(x),
        "Min": x.min(),
        "Max": x.max(),
        "Mean": x.mean(),
        "Range": data_range(x),
        "Variance (sample)": variance(x, True),
        "Variance (population)": variance(x, False),
        "Std Dev (sample)": std_dev(x, True),
        "Std Dev (population)": std_dev(x, False),
        "RMSD (about mean)": rmsd(x),
        "Coeff of Dispersion (quartile)": q_cd,
        "Coeff of Variation (%)": cv,
        "Coeff of Dispersion (MAD/median)": mad_cd,
    }


def main():
    df = load_data(CSV_PATH)
    print(f"Records used: {len(df):,}\n")

    results = {}
    for col in COLUMNS:
        results[col] = compute_all(df[col].values.astype(float))

    table = pd.DataFrame(results)

    pd.set_option("display.width", 250, "display.float_format", lambda v: f"{v:,.6f}")
    print(table.to_string())

    manual_check(df["Vict Age"].values.astype(float))

    out_path = os.path.join(OUT_DIR, "dispersion_measures.csv")
    table.to_csv(out_path)
    print(f"\nSaved -> {out_path}")


def manual_check(x):
    n = len(x)
    mean = x.sum() / n
    sq_dev = ((x - mean) ** 2).sum()
    var_sample = sq_dev / (n - 1)
    print("\nManual verification (Vict Age):")
    print(f"  Range              = {x.max() - x.min():.4f}")
    print(f"  Sample variance    = {var_sample:.4f}")
    print(f"  Std deviation      = {np.sqrt(var_sample):.4f}")
    print(f"  RMSD               = {np.sqrt(sq_dev / n):.4f}")
    print(f"  Coeff of variation = {np.sqrt(var_sample) / mean * 100:.4f}%")


if __name__ == "__main__":
    main()