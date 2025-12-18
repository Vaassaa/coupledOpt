"""
Simple python script that rewrites rain.in data
used as a quick fix for getting dimension right...
Author: Vaclav Steinbach
Date: 18.12.2025
Dissertation
"""
import pandas as pd

# ---CONFIG---
campaign_info = "#campaign 2024-09-08 00:00 2024-09-30 00:00"
picked_column = 'throughfall[m/s]'

# Define column names according to the commented header.
column_names = [
    "time[s]", 
    "throughfall[m/s]"
]

# Read the data from .in file into a DataFrame, skipping initial commented rows.
data = pd.read_csv('rain.in', comment='#', sep='\\s+', header=None, names=column_names)

# Rewrite throughfall column values --> m/10min to m/s
data[picked_column] = data[picked_column]/6

# Save the modified data back to .in file starting from third row (index 2), without index and with a custom format.
with open('rain_fixed.in', 'w') as f:
    # Write header manually
    f.write(campaign_info+"\n")
    f.write('#' + '\t'.join(column_names) + '\n')  # Manually add column names with comment
    data.to_string(f, 
                   header=False, 
                   index=False,
                   float_format=lambda x: f"{x:.10f}") # formating digits

print(f"Column {picked_column} is rewriten!")
