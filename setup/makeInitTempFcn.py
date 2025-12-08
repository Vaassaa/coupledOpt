"""
Python script for the construction of intial temperature function
Author: Vaclav Steinbach
Date: 25.10.2025
Dissertation work
"""

import pandas as pd
import sys
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# --- CONFIG ---
# Mesh values from drutes
z_values = [-1.5, -0.5, -0.23, -0.15, -0.08, 0.0]
# Pick a time region
start_date = datetime(2024, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
end_date   = datetime(2024, 9, 30, 0, 0, 0, tzinfo=timezone.utc)
# tree type: buk, smrk, modrin
tree = sys.argv[1]
# location: loc1, loc2, loc3
loc = sys.argv[2]
dat_fol = "dataIN"
PATH = dat_fol+"/"+tree+"/"
out_FOLD = "optData/"

# --- GEARS --- 
# Read topsoil buk data
FILE = "topsoil_"+tree+"_"+loc+".csv"
topsoil = pd.read_csv(PATH + FILE)

# Group the data by ID
ID_group = topsoil.groupby("ID")
# Display information about each group
# for key, item in ID_group:
    # print(f"Group Key: {key}")
    # print(item.info())

# Get the sensors
topsoil_sen1 = ID_group.get_group(94212703)
topsoil_sen1["DTM"] = pd.to_datetime(topsoil_sen1["DTM"], utc=True)
topsoil_sen1 = topsoil_sen1.set_index("DTM") # set index for slicing
topsoil_sen2 = ID_group.get_group(94212707)
topsoil_sen2["DTM"] = pd.to_datetime(topsoil_sen2["DTM"], utc=True)
topsoil_sen2 = topsoil_sen2.set_index("DTM")
topsoil_sen3 = ID_group.get_group(94212719)
topsoil_sen3["DTM"] = pd.to_datetime(topsoil_sen3["DTM"], utc=True)
topsoil_sen3 = topsoil_sen3.set_index("DTM")

# Slice to time range
topsoil_sen1_sliced = topsoil_sen1[start_date:end_date].reset_index()
topsoil_sen2_sliced = topsoil_sen2[start_date:end_date].reset_index()
topsoil_sen3_sliced = topsoil_sen3[start_date:end_date].reset_index()

# print(topsoil_sen1_sliced.info())
# print(topsoil_sen2_sliced.info())
# print(topsoil_sen3_sliced.info())


# Create new dataframe with averaged sensor values
topsoil_avg = topsoil_sen1_sliced.copy() # create a copy to rewrite
topsoil_avg['T1'] = (topsoil_sen1_sliced['T1'] +
                +   topsoil_sen2_sliced['T1'] +
                +   topsoil_sen3_sliced['T1'])/3
topsoil_avg['T2'] = (topsoil_sen1_sliced['T2'] +
                +   topsoil_sen2_sliced['T2'] +
                +   topsoil_sen3_sliced['T2'])/3

topsoil_avg = topsoil_avg.set_index("DTM")
topsoil_avg = topsoil_avg.resample("10min").mean().interpolate()

# print(topsoil_avg['T1'])
# print(topsoil_avg['T2'])

# Subsoil buk data
FILE = "subsoil_"+tree+"_"+loc+".csv"
subsoil = pd.read_csv(PATH + FILE)

ID_group = subsoil.groupby("ID")
# Display information about each group
for key, item in ID_group:
    print(f"Group Key: {key}")
    print(item.info())
# Get the sensors
subsoil_sen1 = ID_group.get_group(94218455)
subsoil_sen1["DTM"] = pd.to_datetime(subsoil_sen1["DTM"], utc=True)
subsoil_sen1 = subsoil_sen1.set_index("DTM") # set index for slicing
subsoil_sen2 = ID_group.get_group(94218474)
subsoil_sen2["DTM"] = pd.to_datetime(subsoil_sen2["DTM"], utc=True)
subsoil_sen2 = subsoil_sen2.set_index("DTM")
subsoil_sen3 = ID_group.get_group(94218475)
subsoil_sen3["DTM"] = pd.to_datetime(subsoil_sen3["DTM"], utc=True)
subsoil_sen3 = subsoil_sen3.set_index("DTM")

# Slice to time range
subsoil_sen1_sliced = subsoil_sen1[start_date:end_date].reset_index()
subsoil_sen2_sliced = subsoil_sen2[start_date:end_date].reset_index()
subsoil_sen3_sliced = subsoil_sen3[start_date:end_date].reset_index()

# Create new dataframe with averaged sensor values
subsoil_avg = subsoil_sen1_sliced.copy() # create a copy to rewrite
subsoil_avg['T1'] = (subsoil_sen1_sliced['T1'] +
                +   subsoil_sen2_sliced['T1'] +
                +   subsoil_sen3_sliced['T1'])/3
subsoil_avg['T2'] = (subsoil_sen1_sliced['T2'] +
                +   subsoil_sen2_sliced['T2'] +
                +   subsoil_sen3_sliced['T2'])/3


subsoil_avg = subsoil_avg.set_index("DTM")
subsoil_avg = subsoil_avg.resample("10min").mean().interpolate()

print(subsoil_avg['T1'])
print(subsoil_avg['T2'])

# Create intial temp array
T0_values = [
        subsoil_avg['T1'].iloc[0], # -23 cm / -1.5m
        subsoil_avg['T1'].iloc[0], # -23 cm / -0.5m 
        subsoil_avg['T1'].iloc[0], # -23 cm
        subsoil_avg['T2'].iloc[0], # -15 cm
        topsoil_avg['T1'].iloc[0], # -8 cm
        topsoil_avg['T2'].iloc[0]] # 0 cm

# Combine z values and the corresponding T values
intialTemp = pd.DataFrame({
    'z': z_values,
    'T': T0_values})

# Save to a .in file with '#' header / comment in fortran
filename = out_FOLD+'heaticond1D.in'
with open(filename, 'w') as f:
    f.write(f"# campaign {start_date} {end_date}\n")
    f.write("# z[m]\tT[˚C]\n")       
    intialTemp.to_csv(f, sep='\t', index=False, header=False)
