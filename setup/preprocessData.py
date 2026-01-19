import pandas as pd
from datetime import datetime, timezone

start_date = datetime(2024, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
end_date   = datetime(2024, 9, 30, 0, 0, 0, tzinfo=timezone.utc)

def load_and_process(path):

    df = pd.read_csv(path)
    df["DTM"] = pd.to_datetime(df["DTM"], utc=True)

    # --- Split by sensor ID ---
    sensor_dfs = {}
    for sensor_id, g in df.groupby("ID"):
        g = g.sort_values("DTM").set_index("DTM")

        # --- Slice date window per sensor ---
        g = g.loc[start_date:end_date]

        sensor_dfs[sensor_id] = g

    # --- Align all sensors to same timestamps ---
    #    outer join ensures we keep full time coverage
    full = pd.concat(sensor_dfs.values(), axis=1, keys=sensor_dfs.keys())

    # Example: full looks like:
    #   ID1.T1  ID1.T2  ID1.T3  ID1.moisture  ID2.T1 ...
    #   with MultiIndex columns (ID, parameter)
    for sensor_id, g in sensor_dfs.items():
        T_cols = [c for c in g.columns if c in ["T1", "T2", "T3"]]
        print(
            f"Sensor {sensor_id} max T:",
            g[T_cols].max().max()
        )

    # --- Compute averages across sensors ---
    T_cols = [c for c in full.columns if c[1] in ["T1", "T2", "T3"]]
    theta_cols = [c for c in full.columns if c[1] == "moisture"]

    full["T_avg"] = full[T_cols].mean(axis=1)
    print("Max T_avg before resampling:", full["T_avg"].max())
    full["theta_avg"] = full[theta_cols].mean(axis=1)

    # Keep only average columns
    result = full[["T_avg", "theta_avg"]]

    # --- Resample to 10 min ---
    result = result.resample("10min").interpolate()
    print("Max T_avg after resampling:", result["T_avg"].max())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,4))
    for c in T_cols:
        plt.plot(full.index, full[c], color="gray", alpha=0.3)

    plt.plot(full.index, full["T_avg"], color="red", lw=2)
    plt.ylabel("Temperature [°C]")
    plt.title("Sensor temperatures and averaged signal")
    plt.show()

    return result


# -------------------------------
# Process topsoil + subsoil
# -------------------------------
subsoil_res = load_and_process("dataIN/buk/subsoil_buk_loc1.csv")
topsoil_res = load_and_process("dataIN/buk/topsoil_buk_loc1.csv")


# -------------------------------
# Combine into final simulation DF
# -------------------------------
df = pd.DataFrame({
    "T_n8cm"      : topsoil_res["T_avg"],
    "T_n15cm"     : topsoil_res["T_avg"],   # adjust if needed
    "T_n23cm"     : subsoil_res["T_avg"],
    "theta_n8cm" : topsoil_res["theta_avg"], # same as n15 for inverse modeling 
    "theta_n23cm" : subsoil_res["theta_avg"],
})

df["seconds"] = (df.index - start_date).total_seconds()
df = df.reset_index(drop=True)

# Move seconds to the first column
cols = df.columns.tolist()
cols = ["seconds"] + [c for c in cols if c != "seconds"]
df = df[cols]


# Create your output folder if it doesn't exist
import os
os.makedirs("out", exist_ok=True)

# File path
out_file = "setup_out/monitoring.dat"

# Write header manually
with open(out_file, "w") as f:
    f.write(f"# campaign: {start_date} {end_date}\n")
    f.write("# time[s] T_n8cm[˚C] T_n15cm[˚C] T_n23cm[˚C] theta_n8cm[-] theta_n23cm[-]\n")

# Append the data without index or header
df.to_csv(out_file, sep=" ", index=False, header=False, mode="a")
print(df.head())

