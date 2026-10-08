import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import FuncFormatter
from scipy import stats

CSV_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
OUT_DIR = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\distribution_plots"
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


if __name__ == "__main__":
    main()