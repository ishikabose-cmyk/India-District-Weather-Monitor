"""
India District Weather Monitor - terminal menu + matplotlib/seaborn charts.
Libraries used: numpy, pandas, matplotlib, seaborn, scipy (nothing else).

Run:  python india_weather_monitor_pandas.py      (weather-1.csv in the same folder)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

CSV = "weather-1.csv"
RN = ["Severe cold", "Cold", "Normal", "Hot", "Heatwave"]
RCOL = ["#1D3E91", "#4F98E3", "#2E9A69", "#E8941A", "#D83A3A"]
TILES = {"HP": (3, 0), "PB": (2, 1), "CH": (3, 1), "UK": (4, 1), "AR": (9, 1), "RJ": (1, 2), "HR": (2, 2), "DL": (3, 2),
         "UP": (4, 2), "BR": (5, 2), "SK": (6, 2), "AS": (8, 2), "NL": (9, 2), "GJ": (1, 3), "MP": (3, 3), "JH": (5, 3),
         "WB": (6, 3), "ML": (8, 3), "MN": (9, 3), "DD": (1, 4), "MH": (3, 4), "CG": (4, 4), "OD": (5, 4), "TR": (8, 4),
         "MZ": (9, 4), "GA": (2, 5), "KA": (3, 5), "TS": (4, 5), "AP": (5, 5), "LD": (2, 6), "KL": (3, 6), "TN": (4, 6),
         "PY": (5, 6), "AN": (8, 6)}
NM = {"HP": "Himachal Pradesh", "PB": "Punjab", "CH": "Chandigarh", "UK": "Uttarakhand", "AR": "Arunachal Pradesh",
      "RJ": "Rajasthan", "HR": "Haryana", "DL": "Delhi", "UP": "Uttar Pradesh", "BR": "Bihar", "SK": "Sikkim",
      "AS": "Assam", "NL": "Nagaland", "GJ": "Gujarat", "MP": "Madhya Pradesh", "JH": "Jharkhand", "WB": "West Bengal",
      "ML": "Meghalaya", "MN": "Manipur", "DD": "Dadra and Nagar Haveli and Daman and Diu", "MH": "Maharashtra",
      "CG": "Chhattisgarh", "OD": "Odisha", "TR": "Tripura", "MZ": "Mizoram", "GA": "Goa", "KA": "Karnataka",
      "TS": "Telangana", "AP": "Andhra Pradesh", "LD": "Lakshadweep", "KL": "Kerala", "TN": "Tamil Nadu",
      "PY": "Puducherry", "AN": "Andaman and Nicobar Islands"}
# temperature colour scale: cold blue -> teal -> amber -> red
CMAP = sns.blend_palette(["#1D3E91", "#5AA2E8", "#78C8AA", "#EE9F26", "#D83A3A"], as_cmap=True)
NORM = plt.Normalize(-25, 40)
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ data
def risk_level(t):
    return np.select([t < 0, t < 10, t < 35, t < 40], [0, 1, 2, 3], default=4)


def heat_index(t, h):
    f = t * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * f + 10.14333127 * h - .22475541 * f * h - .00683783 * f ** 2
          - .05481717 * h ** 2 + .00122874 * f ** 2 * h + .00085282 * f * h ** 2 - .00000199 * f ** 2 * h ** 2)
    return np.where(t >= 26.7, np.round((hi - 32) * 5 / 9, 1), t)


def wind_chill(t, v):
    wc = 13.12 + .6215 * t - 11.37 * v ** .16 + .3965 * t * v ** .16
    return np.where((t <= 10) & (v > 4.8), np.round(wc, 1), t)


def load(path):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = ["state", "district", "temp", "cond", "hum", "wind", "updated"]
    raw["updated"] = pd.to_datetime(raw["updated"])
    raw = raw.sort_values(["state", "district", "updated"])
    raw["k"] = raw.groupby(["state", "district"]).cumcount()
    n = raw["k"].max() + 1

    temps = raw.pivot(index=["state", "district"], columns="k", values="temp")
    temps.columns = [f"t{i}" for i in range(n)]
    latest = raw.groupby(["state", "district"]).tail(1).set_index(["state", "district"])
    df = temps.join(latest[["cond", "hum", "wind", "updated"]]).reset_index()

    # day labels (an early-morning stamp on the final reading belongs to the previous day)
    stamps = raw[raw["district"] == raw["district"].iloc[0]].sort_values("updated")["updated"].tolist()
    days = [s.normalize() for s in stamps]
    if stamps[-1].hour < 6:
        days[-1] -= pd.Timedelta(days=1)
    dates = [d.strftime("%d %b").lstrip("0") for d in days]

    tc = [f"t{i}" for i in range(n)]
    df["temp"] = df[tc[-1]]
    df["hi"] = heat_index(df["temp"].to_numpy(), df["hum"].to_numpy())
    df["wc"] = wind_chill(df["temp"].to_numpy(), df["wind"].to_numpy())
    df["risk"] = risk_level(df["temp"].to_numpy())
    df["risk_name"] = df["risk"].map(dict(enumerate(RN)))
    return df, tc, dates


try:
    ALL, TC, DATES = load(CSV)
except FileNotFoundError:
    ALL, TC, DATES = load(input(f"Could not find {CSV}. Path to the weather CSV> ").strip().strip('"'))

def cur():
    return ALL


def f1(x):
    return f"{x:.1f}".replace("-", "−")


def label(r):
    return f"{r['district']}, {r['state']}"


def title(s):
    print(f"\n{s}\n" + "-" * len(s))


# ------------------------------------------------------------------ screens
def overview():
    df = cur()
    t = df["temp"]
    r, p = stats.pearsonr(t, df["hum"])
    cold, hot = df.loc[t.idxmin()], df.loc[t.idxmax()]
    title(f"Temperatures across {len(df)} districts: {f1(t.min())}°C to {f1(t.max())}°C")
    print(f"{label(cold)} is the coldest reading and {label(hot)} is the warmest."
          + ("" if (df['risk'] == 4).any() else " No district is in heatwave range."))
    print(f"\n  Mean {f1(t.mean())}°C | Median {f1(t.median())}°C | Std dev {t.std(ddof=0):.2f}°C")
    print(f"  Temperature-humidity correlation r = {r:.3f} (p = {p:.4f})")
    print("\n  Risk levels:  " + " | ".join(f"{RN[i]} {int((df['risk'] == i).sum())}" for i in range(5)))
    print("\n  Daily trend:      " + "".join(f"{d:>9}" for d in DATES))
    for name, fn in (("max", "max"), ("mean", "mean"), ("min", "min")):
        print(f"    {name:<14}" + "".join(f"{f1(v):>9}" for v in df[TC].agg(fn)))

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("How the country looks right now", fontsize=15, fontweight="bold")
    sns.histplot(t, bins=24, kde=True, color="#4F98E3", ax=ax[0, 0])
    ax[0, 0].axvline(t.median(), color="k", ls="--", lw=1)
    ax[0, 0].set(title=f"Districts at each temperature (median {f1(t.median())}°C)", xlabel="Temperature (°C)", ylabel="Districts")
    sns.scatterplot(data=df, x="temp", y="hum", hue="risk_name", hue_order=RN, palette=RCOL, s=35, ax=ax[0, 1])
    ax[0, 1].set(title="Humidity against temperature", xlabel="Temperature (°C)", ylabel="Humidity (%)")
    ax[0, 1].legend(title="Risk", fontsize=8)
    cnt = df["risk_name"].value_counts().reindex(RN, fill_value=0)
    sns.barplot(x=cnt.index, y=cnt.values, hue=cnt.index, palette=RCOL, legend=False, ax=ax[0, 2])
    ax[0, 2].set(title="Districts by risk level", xlabel="", ylabel="Districts")
    for i, v in enumerate(cnt.values):
        ax[0, 2].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    xs = np.arange(len(DATES))
    for fn, c, w in (("max", "#D83A3A", 1.5), ("mean", "black", 3), ("min", "#4F98E3", 1.5)):
        ax[1, 0].plot(xs, df[TC].agg(fn).values, color=c, lw=w, marker="o", label=fn.title())
    ax[1, 0].set(title="Daily temperature trend", ylabel="Temperature (°C)", xticks=xs, xticklabels=DATES)
    ax[1, 0].legend()
    corr = pd.DataFrame({"Temperature": df["temp"], "Humidity": df["hum"], "Wind speed": df["wind"], "Heat index": df["hi"]}).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax[1, 1], cbar=False)
    ax[1, 1].set_title("How the weather variables relate")
    sns.boxplot(data=df, x="risk_name", y="hum", order=[n for n in RN if n in set(df["risk_name"])],
                hue="risk_name", palette=dict(zip(RN, RCOL)), legend=False, ax=ax[1, 2])
    ax[1, 2].set(title="Humidity by risk level", xlabel="", ylabel="Humidity (%)")
    plt.tight_layout()
    plt.show()


def state_means():
    return cur().groupby("state")["temp"].mean().sort_values(ascending=False)


def state_map():
    sm = state_means()
    title("Map of state temperatures")
    fig, ax = plt.subplots(figsize=(11, 8))
    for code, (c, r) in TILES.items():
        if NM[code] in sm.index:
            m = sm[NM[code]]
            col = CMAP(NORM(m))
            ax.add_patch(plt.Rectangle((c, -r), .94, .94, color=col))
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            ax.text(c + .47, -r + .47, f"{code}\n{m:.1f}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#0b1620" if lum > .6 else "white")
    ax.set(xlim=(.8, 10.2), ylim=(-6.2, 1.2), aspect="equal", title="Mean temperature by state / UT (°C)")
    ax.axis("off")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, fraction=.03, pad=.02)
    cb.set_label("°C")
    plt.tight_layout()
    plt.show(block=False)
    print("Tile codes: " + ", ".join(sorted(TILES)))
    while True:
        code = input("state code for details (Enter to go back)> ").strip().upper()
        if not code:
            plt.close("all")
            return
        if code in NM and NM[code] in sm.index:
            state_info(NM[code], sm)
        else:
            print("Unknown code. Codes are the 2-letter tile labels.")


def state_info(s, sm):
    d = cur()[cur()["state"] == s].sort_values("temp")
    print(f"\n{s}: {f1(sm[s])}°C (mean of {len(d)} districts)")
    print(f"Coldest: {d.iloc[0]['district']} ({f1(d.iloc[0]['temp'])}°C)   Warmest: {d.iloc[-1]['district']} ({f1(d.iloc[-1]['temp'])}°C)")
    print(", ".join(f"{n} {RN[i].lower()}" for i, n in d["risk"].value_counts().sort_index().items()))
    plt.figure(figsize=(9, max(3, .3 * len(d) + 1.5)))
    ax = sns.barplot(data=d, y="district", x="temp", hue="district", palette=[CMAP(NORM(v)) for v in d["temp"]], legend=False)
    ax.set(title=f"{s}: district temperatures", xlabel="Temperature (°C)", ylabel="")
    plt.tight_layout()
    plt.show()


def menu():
    actions = {"1": overview, "2": state_map}
    while True:
        print("\n" + "=" * 70)
        print(f"India District Weather Monitor · {DATES[0]}-{DATES[-1]} 2025")
        print("=" * 70)
        print("  1) Overview (stats + 6 charts)\n  2) State map\n  q) Quit")
        print("\nRisk: <0°C severe cold | 0-10 cold | 10-35 normal | 35-40 hot | >=40 heatwave.")
        print("Heat index applies from 26.7°C; wind chill at <=10°C with wind > 4.8 km/h.")
        c = input("\n> ").strip().lower()
        if c in ("q", "quit", "exit"):
            break
        actions.get(c, lambda: None)()


if __name__ == "__main__":
    try:
        menu()
    except (KeyboardInterrupt, EOFError):
        print()
        