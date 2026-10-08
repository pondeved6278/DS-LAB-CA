import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import pandas as pd
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
