import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------- 1. READ THE DATA ----------
raw = pd.read_csv("weather-1.csv", encoding="utf-8-sig")
raw.columns = ["state", "district", "temp", "cond", "hum", "wind", "updated"]  # short names
raw["updated"] = pd.to_datetime(raw["updated"])
raw = raw.sort_values("updated")          # raw = ALL readings (used for history)

# ---------- 2. KEEP ONLY THE LATEST READING OF EACH DISTRICT ----------
df = raw.drop_duplicates(["state", "district"], keep="last")   # df = newest row per district

# ---------- 3. ADD A RISK LEVEL FROM TEMPERATURE ----------
names  = ["Severe cold", "Cold", "Normal", "Hot", "Heatwave"]
colors = ["#1D3E91", "#4F98E3", "#2E9A69", "#E8941A", "#D83A3A"]
edges  = [-100, 0, 10, 35, 40, 100]                  # <0 | 0-10 | 10-35 | 35-40 | 40+
df["risk"] = pd.cut(df["temp"], bins=edges, labels=names, right=False)


# ---------- THE THREE GRAPHS ----------
def show_graphs():
    # Graph 1: histogram
    median = df["temp"].median()
    plt.figure()
    sns.histplot(df["temp"], bins=24, kde=True, color="#4F98E3")
    plt.axvline(median, color="black", linestyle="--", linewidth=1)
    plt.title(f"Districts at each temperature (median {median:.1f}°C)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Districts")

    # Graph 2: scatter (humidity vs temperature)
    plt.figure()
    sns.scatterplot(data=df, x="temp", y="hum", hue="risk",
                    hue_order=names, palette=colors, s=35)
    plt.title("Humidity against temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.legend(title="Risk", fontsize=8)

    # Graph 3: bar chart (districts per risk level)
    counts = df["risk"].value_counts().reindex(names, fill_value=0)
    plt.figure()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=colors, legend=False)
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.title("Districts by risk level")
    plt.xlabel("")
    plt.ylabel("Districts")

    plt.show()


# ---------- FIND A DISTRICT BY NAME ----------
def find_district():
    name = input("District name> ").strip().lower()
    found = df[df["district"].str.lower().str.contains(name)]   # partial match works too

    if name == "" or found.empty:
        print("District not found.")
        return None
    if len(found) == 1:
        return found.iloc[0]

    # same name in more than one state -> let the user choose
    print("Multiple matches:")
    for i, (_, r) in enumerate(found.iterrows()):
        print(f"  {i}) {r['district']}, {r['state']}")
    pick = input("Choose number> ").strip()
    if pick.isdigit() and int(pick) < len(found):
        return found.iloc[int(pick)]
    print("Invalid choice.")
    return None


def history_of(row):
    # all readings of this one district (oldest -> newest)
    return raw[(raw["state"] == row["state"]) & (raw["district"] == row["district"])]


# ---------- FEATURE 1: DETAILS OF ONE DISTRICT ----------
def district_details():
    row = find_district()
    if row is None:
        return
    print(f"\n{row['district']}, {row['state']}")
    print("-" * 30)
    print(f"Temperature : {row['temp']} °C")
    print(f"Condition   : {row['cond']}")
    print(f"Humidity    : {row['hum']} %")
    print(f"Wind        : {row['wind']} km/h")
    print(f"Risk level  : {row['risk']}")

    # how does it rank among all districts? (1 = hottest)
    rank = int((df["temp"] > row["temp"]).sum()) + 1
    print(f"Hotness rank: {rank} of {len(df)} districts")

    hist = history_of(row)
    print("\nPast readings:")
    print(hist[["updated", "temp", "hum", "wind"]].to_string(index=False))

    plt.figure()
    plt.plot(hist["updated"], hist["temp"], marker="o", color="#D83A3A")
    plt.title(f"Temperature trend: {row['district']}")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ---------- FEATURE 2: COMPARE TWO DISTRICTS ----------
def compare_districts():
    print("First district:")
    a = find_district()
    if a is None:
        return
    print("Second district:")
    b = find_district()
    if b is None:
        return

    name_a = f"{a['district']} ({a['state']})"
    name_b = f"{b['district']} ({b['state']})"

    table = pd.DataFrame({
        name_a: [a["temp"], a["hum"], a["wind"], a["cond"], a["risk"]],
        name_b: [b["temp"], b["hum"], b["wind"], b["cond"], b["risk"]],
    }, index=["Temperature (°C)", "Humidity (%)", "Wind (km/h)", "Condition", "Risk"])
    print("\n", table)

    diff = a["temp"] - b["temp"]
    print(f"\n{a['district']} is {abs(diff):.1f}°C {'warmer' if diff > 0 else 'colder'} than {b['district']}.")

    # both temperature trends on one graph
    ha, hb = history_of(a), history_of(b)
    plt.figure()
    plt.plot(ha["updated"], ha["temp"], marker="o", label=name_a)
    plt.plot(hb["updated"], hb["temp"], marker="o", label=name_b)
    plt.title("Temperature comparison")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- MENU ----------
while True:
    print("\n1) Show 3 graphs\n2) District details\n3) Compare two districts\nq) Quit")
    choice = input("> ").strip().lower()
    if choice == "1":
        show_graphs()
    elif choice == "2":
        district_details()
    elif choice == "3":
        compare_districts()
    elif choice == "q":
        break