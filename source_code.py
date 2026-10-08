import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DS_DATA_DIR", r"C:\Users\VED PONDE\OneDrive\Desktop\DS")
CSV_PATH = os.path.join(DATA_DIR, "export.csv")
OUT_DIR = DATA_DIR


def section_1_0():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print(df.head())
    print(df.info())


def section_1_1():
    OUTPUT_PATH = os.path.join(DATA_DIR, "attribute_classification.csv")

    ATTRIBUTES = {
        "DR_NO":           ("Nominal",  "N/A",        "Unique record identifier; arithmetic is meaningless"),
        "Date Rptd":       ("Interval", "Discrete",   "Calendar date (day resolution); differences meaningful, no true zero"),
        "DATE OCC":        ("Interval", "Discrete",   "Calendar date (day resolution); differences meaningful, no true zero"),
        "TIME OCC":        ("Interval", "Discrete",   "24h clock time stored as HHMM; arbitrary zero (midnight)"),
        "AREA":            ("Nominal",  "N/A",        "Numeric code for a police division"),
        "AREA NAME":       ("Nominal",  "N/A",        "Name of the police division"),
        "Rpt Dist No":     ("Nominal",  "N/A",        "Reporting-district code"),
        "Part 1-2":        ("Nominal",  "N/A",        "Binary crime category code (Part 1 / Part 2)"),
        "Crm Cd":          ("Nominal",  "N/A",        "Crime code"),
        "Crm Cd Desc":     ("Nominal",  "N/A",        "Crime description"),
        "Mocodes":         ("Nominal",  "N/A",        "Space-separated modus-operandi codes"),
        "Vict Age":        ("Ratio",    "Discrete",   "Age in whole years; true zero exists"),
        "Vict Sex":        ("Nominal",  "N/A",        "Category (M/F/X/...)"),
        "Vict Descent":    ("Nominal",  "N/A",        "Descent code"),
        "Premis Cd":       ("Nominal",  "N/A",        "Premises code"),
        "Premis Desc":     ("Nominal",  "N/A",        "Premises description"),
        "Weapon Used Cd":  ("Nominal",  "N/A",        "Weapon code"),
        "Weapon Desc":     ("Nominal",  "N/A",        "Weapon description"),
        "Status":          ("Nominal",  "N/A",        "Case status code"),
        "Status Desc":     ("Nominal",  "N/A",        "Case status description"),
        "Crm Cd 1":        ("Nominal",  "N/A",        "Primary crime code"),
        "Crm Cd 2":        ("Nominal",  "N/A",        "Secondary crime code"),
        "Crm Cd 3":        ("Nominal",  "N/A",        "Tertiary crime code"),
        "Crm Cd 4":        ("Nominal",  "N/A",        "Quaternary crime code"),
        "LOCATION":        ("Nominal",  "N/A",        "Street address"),
        "Cross Street":    ("Nominal",  "N/A",        "Cross-street name"),
        "LAT":             ("Interval", "Continuous", "Latitude in degrees; arbitrary zero (equator is a convention)"),
        "LON":             ("Interval", "Continuous", "Longitude in degrees; arbitrary zero (prime meridian)"),
    }

    OPERATIONS = {
        "Nominal":  "=, != ; mode, frequency counts, chi-square, contingency tables",
        "Ordinal":  "Nominal ops + <, > ; median, percentiles, rank correlation",
        "Interval": "Ordinal ops + +, - ; mean, std dev, Pearson correlation (no ratios)",
        "Ratio":    "Interval ops + *, / ; ratios, geometric mean, coefficient of variation",
    }


    def main():
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {CSV_PATH}\n")

        missing = set(df.columns) - set(ATTRIBUTES)
        extra = set(ATTRIBUTES) - set(df.columns)
        if missing:
            print("WARNING - unclassified columns:", sorted(missing))
        if extra:
            print("NOTE - classified columns not in file:", sorted(extra))

        rows = []
        for col in df.columns:
            scale, num_type, reason = ATTRIBUTES.get(col, ("Unknown", "Unknown", "Not classified"))
            s = df[col]
            rows.append({
                "Attribute": col,
                "Pandas dtype": str(s.dtype),
                "Scale": scale,
                "Numeric type": num_type,
                "Unique values": s.nunique(),
                "Missing %": round(s.isna().mean() * 100, 2),
                "Valid operations": OPERATIONS.get(scale, "-"),
                "Reason": reason,
            })

        result = pd.DataFrame(rows)

        pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 100)
        print(result[["Attribute", "Pandas dtype", "Scale", "Numeric type",
                      "Unique values", "Missing %"]].to_string(index=False))

        print("\nCount by measurement scale:")
        print(result["Scale"].value_counts().to_string())

        print("\nValid operations by scale:")
        for k, v in OPERATIONS.items():
            print(f"  {k:<9}: {v}")

        print("\nData-quality notes:")
        age = df["Vict Age"]
        print(f"  Vict Age (Ratio): {(age <= 0).sum():,} rows with age <= 0 (invalid / unknown) "
              f"-> exclude before computing means or ratios")
        print(f"  LAT/LON (Interval): {((df['LAT'] == 0) | (df['LON'] == 0)).sum():,} rows with 0 coordinates "
              f"(placeholder for unknown location)")
        print(f"  Nominal codes stored as numbers (AREA, Crm Cd, Premis Cd, ...): "
              f"do NOT take mean/sum of these.")

        result.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved table -> {OUTPUT_PATH}")


    main()


def section_1_2():
    DATE_FORMAT = "%Y %b %d %I:%M:%S %p"
    VALID_SEX = {"M", "F", "X", "H", "N"}
    VALID_DESCENT = set("ABCDFGHIJKLOPSUVWXZ")
    PLACEHOLDERS = {"", " ", "NA", "N/A", "NULL", "NONE", "NAN", "-", "--", "?", "UNKNOWN"}
    LA_LAT_RANGE = (33.5, 34.9)
    LA_LON_RANGE = (-119.0, -117.6)

    CODE_DESC_PAIRS = [
        ("AREA", "AREA NAME"),
        ("Crm Cd", "Crm Cd Desc"),
        ("Premis Cd", "Premis Desc"),
        ("Weapon Used Cd", "Weapon Desc"),
        ("Status", "Status Desc"),
    ]


    def missing_report(df):
        total = len(df)
        nan_count = df.isna().sum()
        placeholder_count = pd.Series(0, index=df.columns)
        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].dropna().astype(str).str.strip().str.upper()
            placeholder_count[col] = s.isin(PLACEHOLDERS).sum()
        report = pd.DataFrame({
            "Missing (NaN)": nan_count,
            "Placeholder strings": placeholder_count,
        })
        report["Total missing"] = report["Missing (NaN)"] + report["Placeholder strings"]
        report["Missing %"] = (report["Total missing"] / total * 100).round(2)
        return report.sort_values("Missing %", ascending=False)


    def noise_report(df, dates_occ, dates_rptd):
        issues = {}

        age = df["Vict Age"]
        issues["Vict Age <= 0"] = int((age <= 0).sum())
        issues["Vict Age > 100"] = int((age > 100).sum())

        valid_age = age[(age > 0) & (age <= 100)]
        q1, q3 = valid_age.quantile([0.25, 0.75])
        iqr = q3 - q1
        issues["Vict Age IQR outliers"] = int(((valid_age < q1 - 1.5 * iqr) | (valid_age > q3 + 1.5 * iqr)).sum())

        issues["LAT == 0"] = int((df["LAT"] == 0).sum())
        issues["LON == 0"] = int((df["LON"] == 0).sum())
        nonzero = (df["LAT"] != 0) & (df["LON"] != 0)
        outside = nonzero & (
            ~df["LAT"].between(*LA_LAT_RANGE) | ~df["LON"].between(*LA_LON_RANGE)
        )
        issues["Coordinates outside LA bounds"] = int(outside.sum())

        t = df["TIME OCC"]
        issues["TIME OCC outside 0-2359"] = int(((t < 0) | (t > 2359)).sum())
        issues["TIME OCC invalid minutes (>59)"] = int((t % 100 > 59).sum())

        issues["Unparseable DATE OCC"] = int(dates_occ.isna().sum())
        issues["Unparseable Date Rptd"] = int(dates_rptd.isna().sum())
        issues["Reported before occurred"] = int((dates_rptd < dates_occ).sum())
        issues["DATE OCC in future"] = int((dates_occ > pd.Timestamp.today()).sum())
        issues["Date Rptd in future"] = int((dates_rptd > pd.Timestamp.today()).sum())

        return pd.Series(issues, name="Count").to_frame()


    def duplicate_report(df):
        issues = {}
        issues["Fully duplicated rows"] = int(df.duplicated().sum())
        issues["Duplicate DR_NO"] = int(df.duplicated(subset=["DR_NO"]).sum())
        key = ["DATE OCC", "TIME OCC", "AREA", "Crm Cd", "LOCATION", "Vict Age", "Vict Sex"]
        issues["Duplicates on event key (date, time, area, crime, location, victim)"] = int(df.duplicated(subset=key).sum())
        return pd.Series(issues, name="Count").to_frame()


    def inconsistency_report(df):
        issues = {}

        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].dropna().astype(str)
            issues[f"{col}: leading/trailing whitespace"] = int((s != s.str.strip()).sum())
            issues[f"{col}: multiple internal spaces"] = int(s.str.contains(r"\s{2,}", regex=True).sum())

        for col in ["AREA NAME", "Crm Cd Desc", "Premis Desc", "Weapon Desc", "Status Desc", "LOCATION", "Cross Street"]:
            s = df[col].dropna().astype(str).str.strip()
            canonical = s.str.upper().str.replace(r"\s+", " ", regex=True)
            variants = s.groupby(canonical).nunique()
            issues[f"{col}: values with case/spacing variants"] = int((variants > 1).sum())

        sex = df["Vict Sex"].dropna().astype(str).str.strip().str.upper()
        issues["Vict Sex: invalid category"] = int((~sex.isin(VALID_SEX)).sum())

        descent = df["Vict Descent"].dropna().astype(str).str.strip().str.upper()
        issues["Vict Descent: invalid code"] = int((~descent.isin(VALID_DESCENT)).sum())

        for code, desc in CODE_DESC_PAIRS:
            sub = df[[code, desc]].dropna()
            issues[f"{code} -> multiple {desc}"] = int((sub.groupby(code)[desc].nunique() > 1).sum())
            issues[f"{desc} -> multiple {code}"] = int((sub.groupby(desc)[code].nunique() > 1).sum())

        issues["Crm Cd != Crm Cd 1"] = int((df["Crm Cd"] != df["Crm Cd 1"]).sum())
        issues["Part 1-2 not in {1,2}"] = int((~df["Part 1-2"].isin([1, 2])).sum())

        crm_cols = ["Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]
        gap = (df["Crm Cd 2"].isna() & df["Crm Cd 3"].notna()) | (df["Crm Cd 3"].isna() & df["Crm Cd 4"].notna())
        issues["Crm Cd gaps (later code filled, earlier empty)"] = int(gap.sum())

        weapon_mismatch = df["Weapon Used Cd"].notna() != df["Weapon Desc"].notna()
        issues["Weapon code/description presence mismatch"] = int(weapon_mismatch.sum())

        moc = df["Mocodes"].dropna().astype(str)
        issues["Mocodes: non-4-digit tokens"] = int(
            moc.apply(lambda v: any(len(tok) != 4 or not tok.isdigit() for tok in v.split())).sum()
        )

        return pd.Series(issues, name="Count").to_frame()


    def quality_score(df, missing, noise, dup, incons):
        total_cells = df.size
        missing_cells = missing["Total missing"].sum()
        completeness = 100 - missing_cells / total_cells * 100
        uniqueness = 100 - dup.loc["Fully duplicated rows", "Count"] / len(df) * 100
        noisy_rows = noise["Count"].sum() / len(df) * 100
        validity = max(0, 100 - noisy_rows)
        inconsistency_rate = incons["Count"].sum() / len(df) * 100
        consistency = max(0, 100 - inconsistency_rate)
        return pd.Series({
            "Completeness %": round(completeness, 2),
            "Uniqueness %": round(uniqueness, 2),
            "Validity % (noise-adjusted)": round(validity, 2),
            "Consistency % (inconsistency-adjusted)": round(consistency, 2),
        }, name="Score").to_frame()


    def main():
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns\n")

        dates_occ = pd.to_datetime(df["DATE OCC"], format=DATE_FORMAT, errors="coerce")
        dates_rptd = pd.to_datetime(df["Date Rptd"], format=DATE_FORMAT, errors="coerce")

        pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 90)

        missing = missing_report(df)
        print("=" * 70)
        print("1. MISSING DATA")
        print("=" * 70)
        print(missing.to_string())

        noise = noise_report(df, dates_occ, dates_rptd)
        print("\n" + "=" * 70)
        print("2. NOISY / INVALID VALUES")
        print("=" * 70)
        print(noise.to_string())

        dup = duplicate_report(df)
        print("\n" + "=" * 70)
        print("3. DUPLICATES")
        print("=" * 70)
        print(dup.to_string())

        incons = inconsistency_report(df)
        print("\n" + "=" * 70)
        print("4. DATA ENTRY INCONSISTENCIES (non-zero only)")
        print("=" * 70)
        print(incons[incons["Count"] > 0].to_string())

        score = quality_score(df, missing, noise, dup, incons)
        print("\n" + "=" * 70)
        print("5. QUALITY SUMMARY")
        print("=" * 70)
        print(score.to_string())

        missing.to_csv(os.path.join(OUT_DIR, "dq_missing.csv"))
        noise.to_csv(os.path.join(OUT_DIR, "dq_noise.csv"))
        dup.to_csv(os.path.join(OUT_DIR, "dq_duplicates.csv"))
        incons.to_csv(os.path.join(OUT_DIR, "dq_inconsistencies.csv"))
        score.to_csv(os.path.join(OUT_DIR, "dq_summary.csv"))
        print(f"\nReports saved in {OUT_DIR}")


    main()


def section_1_3():
    df = pd.read_csv(CSV_PATH, low_memory=False)

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


def section_1_4():
    import tkinter as tk
    from tkinter import filedialog

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


def section_1_5():
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


    main()


def section_2_1():
    from scipy import stats

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

        out_path = os.path.join(DATA_DIR, "central_tendency.csv")
        comparison.round(4).to_csv(out_path)
        print(f"\nSaved -> {out_path}")


    main()


def section_2_2():
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


    main()


def section_2_3():
    import tkinter as tk
    from tkinter import filedialog

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
    res.to_csv(os.path.join(DATA_DIR, "summary.csv"), index=False)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, c in zip(axes.ravel(), cols):
        ax.boxplot(df[c].dropna(), vert=False)
        ax.set_title(c)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "boxplots.png"))
    plt.show()


def section_2_4():
    import tkinter as tk
    from tkinter import filedialog
    from collections import Counter

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


def section_3_1():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    OUT_DIR = os.path.join(DATA_DIR, "frequency_plots")
    os.makedirs(OUT_DIR, exist_ok=True)

    BG = "#FAFAF7"
    INK = "#2B2D42"
    MUTED = "#8D99AE"
    BLUE = "#3A86FF"
    TEAL = "#06B6A4"
    ORANGE = "#FF8F1F"
    PINK = "#EF476F"
    GOLD = "#FFB703"
    CMAP = LinearSegmentedColormap.from_list("calm", ["#BFD7FF", "#3A86FF", "#1D3FA8"])

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "xtick.color": INK,
        "ytick.color": INK,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    VARIABLES = [
        {
            "key": "age",
            "title": "Age of Crime Victims",
            "xlabel": "Victim age (years)",
            "unit": "yrs",
            "edges": np.arange(0, 105, 5),
            "noun": "victims",
        },
        {
            "key": "hour",
            "title": "Time of Day When Crimes Happen",
            "xlabel": "Hour of the day (0 = midnight, 12 = noon)",
            "unit": "h",
            "edges": np.arange(0, 25, 1),
            "noun": "crimes",
        },
    ]


    def load_data(path):
        df = pd.read_csv(path, usecols=["Vict Age", "TIME OCC"], low_memory=False)
        out = pd.DataFrame()
        out["age"] = df["Vict Age"]
        out["hour"] = df["TIME OCC"] // 100 + (df["TIME OCC"] % 100) / 60
        out = out[(out["age"] > 0) & (out["age"] <= 100)]
        out = out[(out["hour"] >= 0) & (out["hour"] < 24)]
        return out.dropna().reset_index(drop=True)


    def fmt_int(v, _=None):
        return f"{int(v):,}"


    def fmt_count(v, _=None):
        if v >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v / 1_000:.0f}K"
        return f"{int(v)}"


    def fmt_hour(h):
        h = int(round(h)) % 24
        suffix = "AM" if h < 12 else "PM"
        base = h % 12
        return f"{12 if base == 0 else base} {suffix}"


    def describe(v, cfg, x):
        return fmt_hour(x) if cfg["key"] == "hour" else f"{x:.0f} {cfg['unit']}"


    def class_label(cfg, lo, hi):
        if cfg["key"] == "hour":
            return f"{fmt_hour(lo)} to {fmt_hour(hi)}"
        return f"{lo:.0f} to {hi:.0f} yrs"


    def style_axis(ax, cfg, ylabel):
        ax.set_xlabel(cfg["xlabel"], fontsize=11, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_count))
        ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        if cfg["key"] == "hour":
            ticks = np.arange(0, 25, 3)
            ax.set_xticks(ticks)
            ax.set_xticklabels([fmt_hour(t) if t < 24 else "12 AM" for t in ticks])
        else:
            ax.set_xticks(np.arange(0, 105, 10))


    def header(ax, title, subtitle):
        ax.set_title(title, fontsize=15, color=INK, loc="left", pad=34)
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")


    def draw_histogram(ax, x, cfg):
        counts, edges = np.histogram(x, bins=cfg["edges"])
        widths = np.diff(edges)
        norm = counts / counts.max()
        colors = CMAP(0.25 + 0.75 * norm)
        peak = counts.argmax()
        colors[peak] = matplotlib_color(ORANGE)

        ax.bar(edges[:-1], counts, width=widths, align="edge", color=colors,
               edgecolor=BG, linewidth=1.5)

        mean, median = x.mean(), x.median()
        ax.axvline(mean, color=PINK, linestyle="--", linewidth=2)
        ax.axvline(median, color=TEAL, linestyle="-", linewidth=2)

        ymax = counts.max()
        ax.set_ylim(0, ymax * 1.28)
        ax.text(mean, ymax * 1.20, f"Average\n{describe(None, cfg, mean)}", color=PINK,
                fontsize=9.5, fontweight="bold", ha="right" if mean > median else "left", va="top")
        ax.text(median, ymax * 1.20, f"Middle value\n{describe(None, cfg, median)}", color=TEAL,
                fontsize=9.5, fontweight="bold", ha="left" if mean > median else "right", va="top")

        px = edges[peak] + widths[peak] / 2
        ax.annotate(f"Most common\n{class_label(cfg, edges[peak], edges[peak + 1])}\n({counts[peak]:,} {cfg['noun']})",
                    xy=(px, ymax), xytext=(px + (edges[-1] - edges[0]) * 0.18, ymax * 0.82),
                    fontsize=9.5, color=ORANGE, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ORANGE, lw=1.2))

        style_axis(ax, cfg, f"Number of {cfg['noun']}")
        header(ax, f"Histogram: {cfg['title']}",
               "Each bar shows how many fall in that range. Taller bar = more common.")
        return counts, edges


    def matplotlib_color(hex_color):
        from matplotlib.colors import to_rgba
        return to_rgba(hex_color)


    def draw_polygon(ax, x, cfg):
        counts, edges = np.histogram(x, bins=cfg["edges"])
        width = edges[1] - edges[0]
        mids = (edges[:-1] + edges[1:]) / 2
        px = np.concatenate([[mids[0] - width], mids, [mids[-1] + width]])
        py = np.concatenate([[0], counts, [0]])

        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=BLUE,
               alpha=0.10, edgecolor="none")
        ax.fill_between(px, py, color=TEAL, alpha=0.22)
        ax.plot(px, py, color=TEAL, linewidth=3, marker="o", markersize=7,
                markerfacecolor="white", markeredgewidth=2.2, markeredgecolor=TEAL)

        peak = counts.argmax()
        ax.scatter([mids[peak]], [counts[peak]], s=170, color=ORANGE, zorder=5,
                   edgecolor="white", linewidth=2)
        ax.annotate(f"Peak around {describe(None, cfg, mids[peak])}",
                    xy=(mids[peak], counts[peak]),
                    xytext=(mids[peak] + (edges[-1] - edges[0]) * 0.15, counts[peak] * 1.02),
                    fontsize=10, fontweight="bold", color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ORANGE, lw=1.2))

        ax.set_ylim(0, counts.max() * 1.25)
        ax.set_xlim(edges[0] - width * 0.5, edges[-1] + width * 0.5)
        style_axis(ax, cfg, f"Number of {cfg['noun']}")
        header(ax, f"Frequency Polygon: {cfg['title']}",
               "Dots mark the centre of each range, joined by a line. It shows the overall shape.")


    def draw_ogive(ax, x, cfg):
        counts, edges = np.histogram(x, bins=cfg["edges"])
        cum = np.concatenate([[0], np.cumsum(counts)])
        total = cum[-1]
        pct = cum / total * 100

        ax.fill_between(edges, pct, color=BLUE, alpha=0.15)
        ax.plot(edges, pct, color=BLUE, linewidth=3, marker="o", markersize=6,
                markerfacecolor="white", markeredgewidth=2, markeredgecolor=BLUE)

        marks = [(25, "25%", GOLD), (50, "Half (50%)", PINK), (75, "75%", TEAL)]
        for level, label, color in marks:
            xv = np.interp(level, pct, edges)
            ax.hlines(level, edges[0], xv, color=color, linestyle=":", linewidth=1.8)
            ax.vlines(xv, 0, level, color=color, linestyle=":", linewidth=1.8)
            ax.scatter([xv], [level], s=120, color=color, zorder=5, edgecolor="white", linewidth=2)
            ax.annotate(f"{label} are below\n{describe(None, cfg, xv)}",
                        xy=(xv, level), xytext=(xv + (edges[-1] - edges[0]) * 0.04, level - 13),
                        fontsize=9.5, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2))

        ax.set_ylim(0, 105)
        ax.set_xlim(edges[0], edges[-1])
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlabel(cfg["xlabel"], fontsize=11, labelpad=8)
        ax.set_ylabel(f"Share of all {cfg['noun']} (running total)", fontsize=11, labelpad=8)
        ax.grid(color=MUTED, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        if cfg["key"] == "hour":
            ticks = np.arange(0, 25, 3)
            ax.set_xticks(ticks)
            ax.set_xticklabels([fmt_hour(t) if t < 24 else "12 AM" for t in ticks])
        else:
            ax.set_xticks(np.arange(0, 105, 10))

        right = ax.secondary_yaxis("right", functions=(lambda p: p / 100 * total, lambda c: c / total * 100))
        right.yaxis.set_major_formatter(FuncFormatter(fmt_count))
        right.set_ylabel(f"Number of {cfg['noun']} (running total)", fontsize=11, labelpad=8, color=MUTED)
        right.spines["right"].set_visible(True)
        right.spines["right"].set_color(MUTED)

        header(ax, f"Cumulative Frequency Curve (Ogive): {cfg['title']}",
               "Reads like a running total. Pick a value on the bottom and see what share falls below it.")


    def save_single(draw_fn, x, cfg, name):
        fig, ax = plt.subplots(figsize=(11, 6.2))
        draw_fn(ax, x, cfg)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"{cfg['key']}_{name}.png")
        fig.savefig(path, dpi=170, facecolor=BG)
        plt.close(fig)
        return path


    def save_dashboard(x, cfg):
        fig, axes = plt.subplots(1, 3, figsize=(27, 7.6))
        draw_histogram(axes[0], x, cfg)
        draw_polygon(axes[1], x, cfg)
        draw_ogive(axes[2], x, cfg)
        fig.suptitle(f"How Are {cfg['title']} Spread Out?", fontsize=24, fontweight="bold",
                     color=INK, y=1.03)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"{cfg['key']}_dashboard.png")
        fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return path


    def frequency_table(x, cfg):
        counts, edges = np.histogram(x, bins=cfg["edges"])
        table = pd.DataFrame({
            "Class": [class_label(cfg, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])],
            "Lower": edges[:-1],
            "Upper": edges[1:],
            "Midpoint": (edges[:-1] + edges[1:]) / 2,
            "Frequency": counts,
        })
        table["Relative Freq %"] = (table["Frequency"] / table["Frequency"].sum() * 100).round(2)
        table["Cumulative Freq"] = table["Frequency"].cumsum()
        table["Cumulative %"] = (table["Cumulative Freq"] / table["Frequency"].sum() * 100).round(2)
        return table


    def main():
        df = load_data(CSV_PATH)
        print(f"Records used: {len(df):,}\n")

        for cfg in VARIABLES:
            x = df[cfg["key"]]
            print("=" * 70)
            print(cfg["title"])
            print("=" * 70)

            table = frequency_table(x, cfg)
            print(table.to_string(index=False))
            table.to_csv(os.path.join(OUT_DIR, f"{cfg['key']}_frequency_table.csv"), index=False)

            paths = [
                save_single(draw_histogram, x, cfg, "histogram"),
                save_single(draw_polygon, x, cfg, "frequency_polygon"),
                save_single(draw_ogive, x, cfg, "ogive"),
                save_dashboard(x, cfg),
            ]
            print("\nSaved:")
            for p in paths:
                print(" ", p)
            print()


    main()


def section_3_2():
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.ticker import FuncFormatter
    from scipy import stats

    OUT_DIR = os.path.join(DATA_DIR, "distribution_plots")
    os.makedirs(OUT_DIR, exist_ok=True)

    BG = "#FAFAF7"
    INK = "#2B2D42"
    MUTED = "#8D99AE"
    BLUE = "#3A86FF"
    TEAL = "#06B6A4"
    ORANGE = "#FF8F1F"
    PINK = "#EF476F"
    GOLD = "#FFB703"
    CMAP = LinearSegmentedColormap.from_list("dev", ["#06B6A4", "#FFB703", "#EF476F"])

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    RNG = np.random.default_rng(42)


    def fmt_hour(h, _=None):
        h = int(round(h)) % 24
        suffix = "AM" if h < 12 else "PM"
        base = h % 12
        return f"{12 if base == 0 else base} {suffix}"


    VARIABLES = [
        {
            "key": "age",
            "title": "Age of Crime Victims",
            "label": "Victim age (years)",
            "fmt": lambda v, _=None: f"{v:.0f}",
            "short": lambda v: f"{v:.0f} yrs",
            "ticks": np.arange(0, 110, 10),
            "noun": "victims",
            "low_word": "very young",
            "high_word": "very old",
        },
        {
            "key": "hour",
            "title": "Time of Day When Crimes Happen",
            "label": "Time of day",
            "fmt": fmt_hour,
            "short": lambda v: fmt_hour(v),
            "ticks": np.arange(0, 25, 3),
            "noun": "crimes",
            "low_word": "very early",
            "high_word": "very late",
        },
        {
            "key": "lat",
            "title": "North-South Location of Crimes in Los Angeles",
            "label": "Latitude (higher = further north)",
            "fmt": lambda v, _=None: f"{v:.2f}°",
            "short": lambda v: f"{v:.3f}°",
            "ticks": None,
            "noun": "crimes",
            "low_word": "far south",
            "high_word": "far north",
        },
    ]


    def load_data(path):
        df = pd.read_csv(path, usecols=["Vict Age", "TIME OCC", "LAT", "LON"], low_memory=False)
        out = pd.DataFrame()
        out["age"] = df["Vict Age"]
        out["hour"] = df["TIME OCC"] // 100 + (df["TIME OCC"] % 100) / 60
        out["lat"] = df["LAT"]
        out = out[(out["age"] > 0) & (out["age"] <= 100)]
        out = out[(out["hour"] >= 0) & (out["hour"] < 24)]
        out = out[(out["lat"] > 33) & (out["lat"] < 35)]
        return out.dropna().reset_index(drop=True)


    def box_stats(x):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inside = x[(x >= lo_fence) & (x <= hi_fence)]
        return {
            "q1": q1, "median": med, "q3": q3, "iqr": iqr,
            "lo_fence": lo_fence, "hi_fence": hi_fence,
            "whisker_lo": inside.min(), "whisker_hi": inside.max(),
            "n_low": int((x < lo_fence).sum()),
            "n_high": int((x > hi_fence).sum()),
        }


    def normality_info(x):
        sample = x.sample(min(5000, len(x)), random_state=1)
        skew = stats.skew(x)
        kurt = stats.kurtosis(x)
        shapiro_p = stats.shapiro(sample).pvalue
        dag_p = stats.normaltest(sample).pvalue
        probs = (np.arange(1, 401) - 0.5) / 400
        theo = stats.norm.ppf(probs)
        actual = np.quantile(x, probs)
        match = np.corrcoef(theo, actual)[0, 1] ** 2 * 100
        if abs(skew) < 0.5 and abs(kurt) < 1:
            verdict, color = "Looks close to a normal bell curve", TEAL
        elif abs(skew) < 1 and abs(kurt) < 2:
            verdict, color = "Roughly bell-shaped, with some differences", GOLD
        else:
            verdict, color = "Does NOT follow a normal bell curve", PINK
        return {
            "skew": skew, "kurt": kurt, "shapiro_p": shapiro_p, "dagostino_p": dag_p,
            "match": match, "verdict": verdict, "color": color,
            "probs": probs, "theo": theo, "actual": actual,
        }


    def header(ax, title, subtitle):
        ax.set_title(title, fontsize=15, fontweight="bold", color=INK, loc="left", pad=34)
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")


    def skew_sentence(skew, cfg):
        if abs(skew) < 0.3:
            return "Data is fairly balanced on both sides."
        if skew > 0:
            return f"More {cfg['high_word']} values pull the right side out."
        return f"More {cfg['low_word']} values pull the left side out."


    def draw_boxplot(ax, x, cfg):
        s = box_stats(x)
        vals = x.values
        ax.boxplot(
            vals, vert=False, widths=0.42, patch_artist=True, whis=1.5, showfliers=False,
            positions=[1],
            boxprops=dict(facecolor=to_rgba(BLUE, 0.35), edgecolor=BLUE, linewidth=2.2),
            medianprops=dict(color=ORANGE, linewidth=4),
            whiskerprops=dict(color=BLUE, linewidth=2.2),
            capprops=dict(color=BLUE, linewidth=2.2),
        )

        outliers = vals[(vals < s["lo_fence"]) | (vals > s["hi_fence"])]
        if len(outliers):
            shown = RNG.choice(outliers, size=min(1500, len(outliers)), replace=False)
            jitter = 1 + RNG.uniform(-0.17, 0.17, len(shown))
            ax.scatter(shown, jitter, s=22, color=PINK, alpha=0.45, edgecolor="none", zorder=3)

        xr = vals.max() - vals.min()
        ax.set_ylim(0.15, 1.95)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

        def tag_below(v, text, color):
            ax.annotate(text, xy=(v, 0.78), xytext=(v, 0.42), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.3))

        def tag_above(v, text, color):
            ax.annotate(text, xy=(v, 1.0), xytext=(v, 1.55), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.3))

        tag_below(s["q1"], f"25% of {cfg['noun']}\nare below {cfg['short'](s['q1'])}", BLUE)
        tag_below(s["q3"], f"75% of {cfg['noun']}\nare below {cfg['short'](s['q3'])}", BLUE)
        tag_above(s["median"], f"Middle value\n{cfg['short'](s['median'])}", ORANGE)
        tag_above(s["whisker_lo"], f"Lowest typical\n{cfg['short'](s['whisker_lo'])}", MUTED)
        tag_above(s["whisker_hi"], f"Highest typical\n{cfg['short'](s['whisker_hi'])}", MUTED)

        total = len(vals)
        if len(outliers):
            msg = (f"{len(outliers):,} unusual values ({len(outliers) / total * 100:.1f}%)\n"
                   f"{s['n_low']:,} unusually low  |  {s['n_high']:,} unusually high")
            color = PINK
        else:
            msg = "No unusual values found"
            color = TEAL
        ax.text(0.99, 0.04, msg, transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1.5))

        ax.text(0.01, 0.04,
                "How to read: the box holds the middle half of the data.\n"
                "The orange line is the middle value. Pink dots are unusual values.",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=MUTED)

        if cfg["ticks"] is not None:
            ax.set_xticks(cfg["ticks"])
        ax.xaxis.set_major_formatter(FuncFormatter(cfg["fmt"]))
        ax.set_xlim(vals.min() - xr * 0.04, vals.max() + xr * 0.04)
        ax.set_xlabel(cfg["label"], fontsize=11, labelpad=8)
        ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        header(ax, f"Boxplot: {cfg['title']}", "Shows where most data sits and which values are unusually far away.")
        return s


    def draw_qq(ax, x, cfg, info):
        mean, std = x.mean(), x.std()
        expected = mean + std * info["theo"]
        actual = info["actual"]
        dev = np.abs(actual - expected) / std
        lo = min(expected.min(), actual.min())
        hi = max(expected.max(), actual.max())
        pad = (hi - lo) * 0.05

        ax.fill_between([lo - pad, hi + pad], [lo - pad - std * 0.25, hi + pad - std * 0.25],
                        [lo - pad + std * 0.25, hi + pad + std * 0.25], color=TEAL, alpha=0.12, zorder=1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=TEAL, linewidth=2.5,
                linestyle="--", zorder=2, label="Perfect bell curve")
        sc = ax.scatter(expected, actual, c=dev, cmap=CMAP, vmin=0, vmax=max(1.0, dev.max()),
                        s=48, edgecolor="white", linewidth=0.6, zorder=3)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
        cbar.set_label("How far a dot is from the line", fontsize=10, color=INK)
        cbar.outline.set_edgecolor(MUTED)

        for idx, word in [(3, "low"), (len(expected) - 4, "high")]:
            pass
        low_i, high_i = 6, len(expected) - 7
        ax.annotate("Lower end", xy=(expected[low_i], actual[low_i]),
                    xytext=(expected[low_i] + (hi - lo) * 0.10, actual[low_i] - (hi - lo) * 0.10),
                    fontsize=9.5, fontweight="bold", color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=1.2))
        ax.annotate("Upper end", xy=(expected[high_i], actual[high_i]),
                    xytext=(expected[high_i] - (hi - lo) * 0.28, actual[high_i] + (hi - lo) * 0.06),
                    fontsize=9.5, fontweight="bold", color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=1.2))

        verdict = (f"{info['verdict']}\n"
                   f"Bell-curve match: {info['match']:.0f}%\n"
                   f"{skew_sentence(info['skew'], cfg)}")
        ax.text(0.03, 0.97, verdict, transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                fontweight="bold", color=info["color"],
                bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=info["color"], lw=1.8))

        ax.text(0.97, 0.03,
                "How to read: dots on the dashed line = data behaves like a\n"
                "normal bell curve. Dots drifting away = it does not.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=MUTED)

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        if cfg["ticks"] is not None:
            ax.set_xticks(cfg["ticks"])
            ax.set_yticks(cfg["ticks"])
        ax.xaxis.set_major_formatter(FuncFormatter(cfg["fmt"]))
        ax.yaxis.set_major_formatter(FuncFormatter(cfg["fmt"]))
        ax.set_xlabel(f"Expected if data were a perfect bell curve", fontsize=11, labelpad=8)
        ax.set_ylabel(f"Actual: {cfg['label'].lower()}", fontsize=11, labelpad=8)
        ax.grid(color=MUTED, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        header(ax, f"Q-Q Plot: {cfg['title']}", "Checks whether the data follows the familiar bell-curve (normal) pattern.")


    def save_single(fn, x, cfg, name, *extra):
        fig, ax = plt.subplots(figsize=(11.5, 6.4))
        fn(ax, x, cfg, *extra)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"{cfg['key']}_{name}.png")
        fig.savefig(path, dpi=170, facecolor=BG)
        plt.close(fig)
        return path


    def save_dashboard(x, cfg, info):
        fig, axes = plt.subplots(1, 2, figsize=(23, 7.8), gridspec_kw={"width_ratios": [1.15, 1]})
        draw_boxplot(axes[0], x, cfg)
        draw_qq(axes[1], x, cfg, info)
        fig.suptitle(f"Spread, Outliers and Normality: {cfg['title']}", fontsize=22,
                     fontweight="bold", color=INK, y=1.04)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"{cfg['key']}_dashboard.png")
        fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return path


    def save_comparison_boxplot(df):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        colors = [BLUE, TEAL, ORANGE]
        for ax, cfg, color in zip(axes, VARIABLES, colors):
            x = df[cfg["key"]].values
            ax.boxplot(x, vert=True, widths=0.55, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=to_rgba(color, 0.35), edgecolor=color, linewidth=2.2),
                       medianprops=dict(color=INK, linewidth=3.5),
                       whiskerprops=dict(color=color, linewidth=2.2),
                       capprops=dict(color=color, linewidth=2.2))
            s = box_stats(pd.Series(x))
            out = x[(x < s["lo_fence"]) | (x > s["hi_fence"])]
            if len(out):
                shown = RNG.choice(out, size=min(1200, len(out)), replace=False)
                ax.scatter(1 + RNG.uniform(-0.15, 0.15, len(shown)), shown, s=14,
                           color=PINK, alpha=0.4, edgecolor="none")
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(FuncFormatter(cfg["fmt"]))
            if cfg["ticks"] is not None:
                ax.set_yticks(cfg["ticks"])
            ax.set_title(cfg["title"], fontsize=12.5, fontweight="bold", color=INK, pad=12)
            ax.text(0.5, -0.06, f"Unusual values: {len(out):,} ({len(out) / len(x) * 100:.1f}%)",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                    fontweight="bold", color=PINK if len(out) else TEAL)
            ax.grid(axis="y", color=MUTED, alpha=0.25)
            ax.set_axisbelow(True)
        fig.suptitle("Side-by-Side: Where Is the Middle and What Is Unusual?", fontsize=20,
                     fontweight="bold", color=INK, y=1.03)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, "all_boxplots_comparison.png")
        fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return path


    def main():
        df = load_data(CSV_PATH)
        print(f"Records used: {len(df):,}\n")

        summary = []
        for cfg in VARIABLES:
            x = df[cfg["key"]]
            s = box_stats(x)
            info = normality_info(x)

            print("=" * 70)
            print(cfg["title"])
            print("=" * 70)
            print(f"Q1 = {s['q1']:.3f} | Median = {s['median']:.3f} | Q3 = {s['q3']:.3f} | IQR = {s['iqr']:.3f}")
            print(f"Outlier fences: [{s['lo_fence']:.3f}, {s['hi_fence']:.3f}]")
            print(f"Outliers: {s['n_low']:,} low, {s['n_high']:,} high")
            print(f"Skewness = {info['skew']:.3f} | Excess kurtosis = {info['kurt']:.3f}")
            print(f"Shapiro-Wilk p = {info['shapiro_p']:.4g} | D'Agostino p = {info['dagostino_p']:.4g}")
            print(f"Bell-curve match = {info['match']:.1f}% -> {info['verdict']}\n")

            paths = [
                save_single(draw_boxplot, x, cfg, "boxplot"),
                save_single(draw_qq, x, cfg, "qqplot", info),
                save_dashboard(x, cfg, info),
            ]
            for p in paths:
                print("Saved:", p)
            print()

            summary.append({
                "Variable": cfg["title"],
                "Q1": s["q1"], "Median": s["median"], "Q3": s["q3"], "IQR": s["iqr"],
                "Lower Fence": s["lo_fence"], "Upper Fence": s["hi_fence"],
                "Low Outliers": s["n_low"], "High Outliers": s["n_high"],
                "Skewness": info["skew"], "Excess Kurtosis": info["kurt"],
                "Shapiro p": info["shapiro_p"], "D'Agostino p": info["dagostino_p"],
                "Bell-Curve Match %": info["match"], "Verdict": info["verdict"],
            })

        print("Saved:", save_comparison_boxplot(df))
        pd.DataFrame(summary).round(4).to_csv(os.path.join(OUT_DIR, "distribution_summary.csv"), index=False)
        print(f"\nAll outputs saved in {OUT_DIR}")


    main()


def section_3_3():
    import seaborn as sns

    df = pd.read_csv(CSV_PATH)

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
    plt.savefig(os.path.join(DATA_DIR, "correlation_heatmap.png"))
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
    plt.savefig(os.path.join(DATA_DIR, "scatter_plots.png"))
    plt.show()

    sns.pairplot(sample, corner=True, plot_kws={"s": 5, "alpha": 0.4})
    plt.savefig(os.path.join(DATA_DIR, "pairplot.png"))
    plt.show()


def section_3_4():
    import tkinter as tk
    from tkinter import filedialog, ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if not path:
        raise SystemExit

    df = pd.read_csv(path, usecols=["DATE OCC", "TIME OCC", "AREA NAME", "Part 1-2",
                                    "Crm Cd Desc", "Vict Age", "Weapon Desc"])
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], format="%Y %b %d %I:%M:%S %p")
    df["year"] = df["DATE OCC"].dt.year
    df["month"] = df["DATE OCC"].dt.to_period("M").dt.to_timestamp()
    df["hour"] = df["TIME OCC"] // 100
    df["age"] = np.where(df["Vict Age"] > 0, df["Vict Age"], np.nan)
    df["weapon"] = df["Weapon Desc"].notna() & ~df["Weapon Desc"].str.startswith("STRONG-ARM", na=False)
    df["part1"] = df["Part 1-2"].eq(1)

    state = {"crime": None, "top": []}
    KPIS = ["Total Incidents", "Serious (Part 1)", "Weapon Involved", "Median Victim Age", "Top Crime"]

    root.title("Crime Scoreboard and Dashboard")
    root.geometry("1200x720")

    bar = ttk.Frame(root, padding=8)
    bar.pack(fill="x")
    area_var, year_var = tk.StringVar(value="All"), tk.StringVar(value="All")

    ttk.Label(bar, text="Area").pack(side="left")
    area_cb = ttk.Combobox(bar, textvariable=area_var, state="readonly", width=16,
                           values=["All"] + sorted(df["AREA NAME"].unique()))
    area_cb.pack(side="left", padx=(4, 12))

    ttk.Label(bar, text="Year").pack(side="left")
    year_cb = ttk.Combobox(bar, textvariable=year_var, state="readonly", width=8,
                           values=["All"] + [str(y) for y in sorted(df["year"].unique())])
    year_cb.pack(side="left", padx=(4, 12))

    crime_lbl = ttk.Label(bar, text="Crime: All")

    board = ttk.Frame(root, padding=8)
    board.pack(fill="x")
    vals = {}
    for k in KPIS:
        box = ttk.LabelFrame(board, text=k, padding=8)
        box.pack(side="left", expand=True, fill="both", padx=4)
        vals[k] = tk.StringVar(value="-")
        ttk.Label(box, textvariable=vals[k], font=("Arial", 14), wraplength=190).pack()

    fig = Figure(figsize=(12, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax1, ax2, ax3 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)


    def update():
        ctx = df
        if area_var.get() != "All":
            ctx = ctx[ctx["AREA NAME"] == area_var.get()]
        if year_var.get() != "All":
            ctx = ctx[ctx["year"] == int(year_var.get())]
        f = ctx if state["crime"] is None else ctx[ctx["Crm Cd Desc"] == state["crime"]]
        n = len(f)

        vals["Total Incidents"].set(f"{n:,}")
        vals["Serious (Part 1)"].set(f"{f['part1'].mean():.1%}" if n else "-")
        vals["Weapon Involved"].set(f"{f['weapon'].mean():.1%}" if n else "-")
        vals["Median Victim Age"].set(f"{f['age'].median():.0f}" if n else "-")
        vals["Top Crime"].set(f["Crm Cd Desc"].value_counts().idxmax() if n else "-")
        crime_lbl.config(text=f"Crime: {state['crime'] or 'All'}")

        top = ctx["Crm Cd Desc"].value_counts().head(10)
        state["top"] = top.index.tolist()

        for ax in (ax1, ax2, ax3):
            ax.clear()

        ax1.barh(range(len(top)), top.values,
                 color=["tab:orange" if c == state["crime"] else "tab:blue" for c in top.index])
        ax1.set_yticks(range(len(top)), [c[:30] for c in top.index], fontsize=8)
        ax1.invert_yaxis()
        ax1.set_title("Top 10 crimes (click a bar to drill down)", fontsize=10)

        hourly = f["hour"].value_counts().reindex(range(24), fill_value=0)
        ax2.bar(hourly.index, hourly.values)
        ax2.set_title("Incidents by hour", fontsize=10)

        trend = f.groupby("month").size()
        ax3.plot(trend.index, trend.values)
        ax3.set_title("Incidents by month", fontsize=10)

        canvas.draw()


    def on_click(event):
        if event.inaxes is ax1 and event.ydata is not None:
            i = int(round(event.ydata))
            if 0 <= i < len(state["top"]):
                c = state["top"][i]
                state["crime"] = None if state["crime"] == c else c
                update()


    def reset():
        area_var.set("All")
        year_var.set("All")
        state["crime"] = None
        update()


    ttk.Button(bar, text="Reset", command=reset).pack(side="left", padx=(0, 12))
    crime_lbl.pack(side="left")

    for cb in (area_cb, year_cb):
        cb.bind("<<ComboboxSelected>>", lambda e: update())
    canvas.mpl_connect("button_press_event", on_click)

    root.deiconify()
    update()
    root.mainloop()


SECTIONS = {
    "1.0": ("Dataset Overview", section_1_0),
    "1.1": ("Attribute Identification", section_1_1),
    "1.2": ("Data Quality Assessment", section_1_2),
    "1.3": ("Missing Data Handling", section_1_3),
    "1.4": ("Data Transformation & Normalization", section_1_4),
    "1.5": ("Discretization & Binning", section_1_5),
    "2.1": ("Measures of Central Tendency", section_2_1),
    "2.2": ("Measures of Dispersion", section_2_2),
    "2.3": ("Shape Measures & Outlier Analysis", section_2_3),
    "2.4": ("Distance & Proximity Calculations", section_2_4),
    "3.1": ("Frequency Distributions", section_3_1),
    "3.2": ("Distribution & Outlier Inspection", section_3_2),
    "3.3": ("Relational & Bivariate Plots", section_3_3),
    "3.4": ("Dashboarding vs. Scoreboards", section_3_4),
}


def run_section(key):
    title, fn = SECTIONS[key]
    print("\n" + "#" * 70)
    print(f"# {key}  {title}")
    print("#" * 70 + "\n")
    plt.rcdefaults()
    try:
        fn()
    except SystemExit:
        print(f"Section {key} cancelled.")
    except Exception as exc:
        print(f"Section {key} failed: {type(exc).__name__}: {exc}")
    finally:
        plt.close("all")


def parse_selection(tokens):
    if not tokens or tokens[0].lower() == "all":
        return list(SECTIONS)
    chosen = []
    for t in tokens:
        t = t.strip()
        if t in SECTIONS:
            chosen.append(t)
        elif t.isdigit():
            chosen.extend(k for k in SECTIONS if k.startswith(t + "."))
        else:
            print(f"Unknown section: {t}")
    return chosen


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tokens = sys.argv[1:]
    if not tokens:
        print("Available sections:")
        for k, (t, _) in SECTIONS.items():
            print(f"  {k}  {t}")
        raw = input("\nEnter section numbers (e.g. 1.1 2.3), a phase number (e.g. 3), or 'all': ")
        tokens = raw.replace(",", " ").split()
    for key in parse_selection(tokens):
        run_section(key)


if __name__ == "__main__":
    main()
