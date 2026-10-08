import os
import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
OUT_DIR = r"C:\Users\VED PONDE\OneDrive\Desktop\DS"

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


if __name__ == "__main__":
    main()