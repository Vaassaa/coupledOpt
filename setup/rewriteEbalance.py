"""
Simple python script that rewrites arbitrary column of ebalance.in
Author: Vaclav Steinbach
Date: 18.01.2026
Dissertation
"""
import pandas as pd

# ---CONFIG---
campaign_info = "#campaign 2024-09-08 00:00 2024-09-30 00:00"
picked_column = 'T_2m[-](era5)'
picked_column = 'T_15cm[°C](amalie)'

# Define column names according to the commented header.
# column_names = [
    # "time[s]", 
    # "S_t[W/m2]", 
    # "T_2m[°C](era5)", 
    # "wind_speed[m/s](era5)", 
    # "total_cloud_cover[-](era5)", 
    # "relative_humidity[%/100](era5)"
# ]
column_names = [
    "time[s]", 
    "S_t[W/m2]", 
    "T_15cm[°C](amalie)", 
    "wind_speed[m/s](era5)", 
    "total_cloud_cover[-](era5)", 
    "relative_humidity[%/100](era5)"
]

# Read the data from .in file into a DataFrame, skipping initial commented rows.
data = pd.read_csv('setup_out/ebalance.in', comment='#', sep='\\s+', header=None, names=column_names)

# Read new data 
new_data = pd.read_csv('setup_out/temp.dat', comment='#', sep='\\s+', header=None, names=["sim_time[s]","temp[˚C]"])

# Rewrite total_cloud_cover column values to 0.8.
data[picked_column] = new_data["temp[˚C]"]

# Save the modified data back to .in file starting from third row (index 2), without index and with a custom format.
with open('setup_out/ebalance.in', 'w') as f:
    # Write header manually
    f.write(campaign_info+"\n")
    f.write('#' + '\t'.join(column_names) + '\n')  # Manually add column names with comment
    data.to_string(f, header=False, index=False)

print(f"Column {picked_column} is rewriten!")
