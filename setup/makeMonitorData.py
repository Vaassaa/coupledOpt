import pandas as pd
from datetime import datetime, timezone
import sys

tree = sys.argv[1]
loc = sys.argv[2]

start_date = datetime(2024, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
end_date   = datetime(2024, 9, 30, 0, 0, 0, tzinfo=timezone.utc)
def process_subsoissl(path):

    df = pd.read_csv(path)
    df["DTM"] = pd.to_datetime(df["DTM"], utc=True)

    sensors = []

    for _, g in df.groupby("ID"):
        g = g.sort_values("DTM").set_index("DTM")
        g = g.loc[start_date:end_date]

        g = g[~g.index.duplicated(keep="first")]  # ← add this line
        g = g.resample("10min").interpolate()
        sensors.append(g)

    full = pd.concat(sensors)

    result = pd.DataFrame(index=full.index.unique().sort_values())

    # −230 mm → T1
    result["T_n23cm"] = (
        full.groupby(full.index)["T1"].mean()
    )

    # −150 mm → T2
    result["T_n15cm"] = (
        full.groupby(full.index)["T2"].mean()
    )

    result["theta_n23cm"] = (
        full.groupby(full.index)["moisture"].mean()

    )

    return result

def process_topsoil_debug(path):
    df = pd.read_csv(path)
    df["DTM"] = pd.to_datetime(df["DTM"], utc=True)
    sensors = []
    for sensor_id, g in df.groupby("ID"):
        g = g.sort_values("DTM").set_index("DTM")
        g = g.loc[start_date:end_date]
        
        # DEBUG
        dups = g.index[g.index.duplicated()]
        if len(dups) > 0:
            print(f"Sensor {sensor_id} has {len(dups)} duplicates BEFORE drop:")
            print(dups)
        
        g = g[~g.index.duplicated(keep="first")]
        
        # DEBUG - confirm gone
        dups_after = g.index[g.index.duplicated()]
        print(f"Sensor {sensor_id}: {len(dups_after)} duplicates after drop, index monotonic: {g.index.is_monotonic_increasing}")
        
        g = g.resample("10min").interpolate()
        sensors.append(g)

def process_topsoilss(path):

    df = pd.read_csv(path)
    df["DTM"] = pd.to_datetime(df["DTM"], utc=True)

    sensors = []

    for _, g in df.groupby("ID"):
        g = g.sort_values("DTM").set_index("DTM")
        g = g.loc[start_date:end_date]

        # resample per sensor
        g = g[~g.index.duplicated(keep="first")]  # ← add this line
        g = g.resample("10min").interpolate()

        sensors.append(g)

    full = pd.concat(sensors)

    result = pd.DataFrame(index=full.index.unique().sort_values())

    # −80 mm → T1
    result["T_n8cm"] = (
        full.groupby(full.index)["T1"].mean()
    )

    # 0 mm → T2
    result["T_n0cm"] = (
        full.groupby(full.index)["T2"].mean()
    )

    # 150 mm → T2
    result["T_15cm"] = (
        full.groupby(full.index)["T3"].mean()
    )

    result["theta_n8cm"] = (
        full.groupby(full.index)["moisture"].mean()
    )

    return result

def process_topsoil(path):
    df = pd.read_csv(path)
    df["DTM"] = pd.to_datetime(df["DTM"], utc=True)
    sensors = []
    for _, g in df.groupby("ID"):
        g = g.sort_values("DTM").set_index("DTM")
        g = g.loc[start_date:end_date]
        g = g[~g.index.duplicated(keep="first")]
        sensors.append(g)  # ← no resample here

    full = pd.concat(sensors)

    # Average across sensors at each timestamp, then resample once
    full = full.groupby(full.index).mean()  # collapse overlapping sensor readings
    full = full.resample("10min").interpolate()  # now index is clean and unique

    result = pd.DataFrame(index=full.index)
    result["T_n8cm"]    = full["T1"]
    result["T_n0cm"]    = full["T2"]
    result["T_15cm"]    = full["T3"]
    result["theta_n8cm"] = full["moisture"]
    return result

def process_subsoil(path):
    df = pd.read_csv(path)
    df["DTM"] = pd.to_datetime(df["DTM"], utc=True)
    sensors = []
    for _, g in df.groupby("ID"):
        g = g.sort_values("DTM").set_index("DTM")
        g = g.loc[start_date:end_date]
        g = g[~g.index.duplicated(keep="first")]
        sensors.append(g)  # ← no resample here

    full = pd.concat(sensors)

    # Average across sensors at each timestamp, then resample once
    full = full.groupby(full.index).mean()  # collapse overlapping sensor readings
    full = full.resample("10min").interpolate()  # now index is clean and unique

    result = pd.DataFrame(index=full.index)
    result["T_n23cm"]    = full["T1"]
    result["T_n15cm"]    = full["T2"]
    result["theta_n23cm"] = full["moisture"]

    return result

top = process_topsoil("dataIN/"+tree+"/topsoil_"+tree+"_"+loc+".csv")
sub = process_subsoil("dataIN/"+tree+"/subsoil_"+tree+"_"+loc+".csv")

df = pd.concat([top, sub], axis=1)

df["seconds"] = (df.index - start_date).total_seconds()
df = df.reset_index(drop=True)
monitoring_data = pd.DataFrame({
    "sim_time[s]" : df["seconds"],
    "T_n8cm"      : df["T_n8cm"],
    "T_n15cm"     : df["T_n15cm"],   # adjust if needed
    "T_n23cm"     : df["T_n23cm"],
    "theta_n8cm"  : df["theta_n8cm"], # same as n15 for inverse modeling 
    "theta_n23cm" : df["theta_n23cm"],
    })

# File path
out_file = "setup_out/"+tree+"_"+loc+"monitoring.dat"

# Write header manually
with open(out_file, "w") as f:
    f.write(f"# campaign: {start_date} {end_date}\n")
    f.write("# time[s] T_n8cm[˚C] T_n15cm[˚C] T_n23cm[˚C] theta_n8cm[-] theta_n23cm[-]\n")

# Append the data without index or header
monitoring_data.to_csv(out_file, sep=" ", index=False, header=False, mode="a")
print(monitoring_data.head())

# Temp above the ground
temp_15cm = pd.DataFrame({
    "sim_time[s]" : df["seconds"],
    "T_15cm"      : df["T_15cm"],
    })

# File path
out_file = "setup_out/"+tree+"_"+loc+"_temp.dat"

# Write header manually
with open(out_file, "w") as f:
    f.write(f"# campaign: {start_date} {end_date}\n")
    f.write("# time[s] T_15cm[˚C]\n")

# Append the data without index or header
temp_15cm.to_csv(out_file, sep=" ", index=False, header=False, mode="a")
print(temp_15cm.head())


