"""
Python script that constructs file ebalance.in used in DRUtES
evaporation module from given meteorological variables
Author: Vaclav Steinbach
Date: 19.11.2025
Dissertation
"""
import pandas as pd

# --- CONFIG ---
campaign = "Campaign_08-09-2024_30-09-2024/"
data_dir = "dataIN/meteo/"+campaign
out_dir = "out/"
# Header info
event_start = "2024-09-08 00:00"
event_end   = "2024-09-30 00:00"

# --- GEARS ---
# Load data from each file
solar = pd.read_csv(data_dir+"solar.in", comment="#", sep=" ", names=["time", "S_t"])
temp = pd.read_csv(data_dir+"temp_interp.in", comment="#", sep=" ", names=["time", "T_a"])
wind = pd.read_csv(data_dir+"wind_interp.in", comment="#", sep=" ", names=["time", "wind_speed"])
cloud = pd.read_csv(data_dir+"clouds_interp.in", comment="#", sep=" ", names=["time", "cloud_cover"])
rh = pd.read_csv(data_dir+"rh_interp.in", comment="#", sep=" ", names=["time", "relative_humidity"])

# Merge all dataframes on 'time'
df = solar.merge(temp, on="time") \
          .merge(wind, on="time") \
          .merge(cloud, on="time") \
          .merge(rh, on="time")

# Write output
with open(out_dir+"ebalance.in", "w") as out:
    out.write(f"#campaign {event_start} {event_end}\n")
    out.write("#time[s]\tS_t[W/m2]\tT_2m[°C](era5)\twind_speed[m/s](era5)\ttotal_cloud_cover[-](era5)\trelative_humidity[%/100](era5)\n")
    df.to_csv(out, sep="\t", index=False, float_format="%.6f", header=False)

print("Saved combined time series to ebalance.in")
