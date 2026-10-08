import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter

CSV_PATH = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\export.csv"
OUT_DIR = r"C:\Users\VED PONDE\OneDrive\Desktop\DS\frequency_plots"
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


if __name__ == "__main__":
    main()