
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. READ WEATHER DATA
raw = pd.read_csv("weather-1.csv", encoding="utf-8-sig")

raw.columns = ["state", "district", "temp", "cond", "hum", "wind", "updated"]

raw["updated"] = pd.to_datetime(raw["updated"])
raw = raw.sort_values("updated")

# Keep latest reading of each district
df = raw.drop_duplicates(["state", "district"], keep="last").copy()

# 2. ADD TEMPERATURE RISK
names = ["Severe cold", "Cold", "Normal", "Hot", "Heatwave"]

colors = ["#1D3E91", "#4F98E3", "#2E9A69", "#E8941A", "#D83A3A"]

edges = [-100, 0, 10, 35, 40, 100]

df["risk"] = pd.cut(df["temp"], bins=edges, labels=names, right=False)


# 3. MODULE 1: SHOW THREE GRAPHS
def show_graphs():

    # Graph 1: Temperature histogram
    median = df["temp"].median()

    plt.figure()
    sns.histplot(df["temp"], bins=24, kde=True)
    plt.axvline(median, color="red", linestyle="--")

    plt.title("Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Number of Districts")

    # Graph 2: Humidity vs Temperature
    plt.figure()
    sns.scatterplot(data=df, x="temp", y="hum",
                    hue="risk", palette=colors)

    plt.title("Humidity vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")

    # Graph 3: Districts in each risk category
    counts = df["risk"].value_counts().reindex(names, fill_value=0)

    plt.figure()
    sns.barplot(x=names, y=counts.values, hue=names,
                palette=colors, legend=False)

    plt.title("Districts by Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Districts")

    plt.show()


# 4. FIND DISTRICT
def find_district():

    name = input("Enter district name: ").strip().lower()

    found = df[df["district"].str.lower().str.contains(name, na=False)]

    if name == "" or found.empty:
        print("District not found.")
        return None

    if len(found) > 1:
        print("Multiple districts found:")

        for i, row in found.iterrows():
            print(i, row["district"], "-", row["state"])

        choice = input("Enter index number: ")

        if choice.isdigit() and int(choice) in found.index:
            return found.loc[int(choice)]

        print("Invalid choice.")
        return None

    return found.iloc[0]


# 5. GET DISTRICT HISTORY
def history_of(row):

    history = raw[
        (raw["state"] == row["state"]) &
        (raw["district"] == row["district"])
    ]

    return history


# 6. MODULE 2: DISTRICT DETAILS
def district_details():

    row = find_district()

    if row is None:
        return

    print("\nDistrict:", row["district"])
    print("State:", row["state"])
    print("Temperature:", row["temp"], "°C")
    print("Condition:", row["cond"])
    print("Humidity:", row["hum"], "%")
    print("Wind:", row["wind"], "km/h")
    print("Risk:", row["risk"])

    # Find temperature rank
    rank = (df["temp"] > row["temp"]).sum() + 1

    print("Hotness Rank:", rank, "out of", len(df))

    # Show past readings
    hist = history_of(row)

    print("\nPast Weather Readings:")
    print(hist[["updated", "temp", "hum", "wind"]])

    # Temperature history graph
    plt.figure()
    plt.plot(hist["updated"], hist["temp"], marker="o")

    plt.title("Temperature History")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 7. MODULE 3: COMPARE TWO DISTRICTS
def compare_districts():

    print("\nEnter First District:")
    a = find_district()

    if a is None:
        return

    print("\nEnter Second District:")
    b = find_district()

    if b is None:
        return

    # Compare weather details
    comparison = pd.DataFrame({

        a["district"]: [a["temp"], a["hum"], a["wind"], a["cond"], a["risk"]],

        b["district"]: [b["temp"], b["hum"], b["wind"], b["cond"], b["risk"]]

    }, index=["Temperature", "Humidity", "Wind", "Condition", "Risk"])

    print("\nWeather Comparison:")
    print(comparison)

    # Calculate temperature difference
    diff = a["temp"] - b["temp"]

    if diff > 0:
        print(a["district"], "is", round(diff, 1), "°C warmer than", b["district"])

    elif diff < 0:
        print(b["district"], "is", round(abs(diff), 1), "°C warmer than", a["district"])

    else:
        print("Both districts have the same temperature.")

    # Compare historical temperatures
    h1 = history_of(a)
    h2 = history_of(b)

    plt.figure()

    plt.plot(h1["updated"], h1["temp"], marker="o",
             label=a["district"])

    plt.plot(h2["updated"], h2["temp"], marker="o",
             label=b["district"])

    plt.title("Temperature Comparison")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 8. MAIN MENU
while True:

    print("\n----- WEATHER ANALYSIS SYSTEM -----")
    print("1. Show Three Graphs")
    print("2. District Details")
    print("3. Compare Two Districts")
    print("q. Quit")

    choice = input("Enter your choice: ").lower().strip()

    if choice == "1":
        show_graphs()

    elif choice == "2":
        district_details()

    elif choice == "3":
        compare_districts()

    elif choice == "q":
        print("Thank you!")
        break

    else:
        print("Invalid choice. Try again.")
