import sys
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
OUTPUT_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\attribute_classification.csv"

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


if __name__ == "__main__":
    main()