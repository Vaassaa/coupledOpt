"""
Python script that makes dirichlet boundary condition
input file for temperature module of DRUtES
Author: Vaclav Steinbach
Date: 26.11.2025
Dissertation
"""
import pandas as pd
from datetime import datetime, timezone

# --- CONFIG ---
start_date = datetime(2024, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
end_date   = datetime(2024, 9, 30, 0, 0, 0, tzinfo=timezone.utc)

# --- GEARS ---
tree = 'buk'    # Example tree type
loc = 'loc1'    # Example location

dat_fol = "dataIN"
PATH = dat_fol + "/" + tree + "/"
out_FOLD = "optData/"

FILE = "topsoil_" + tree + "_" + loc + ".csv"

# Read topsoil data
topsoil = pd.read_csv(PATH + FILE)

# Group the data by ID
ID_group = topsoil.groupby("ID")

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

# Create new dataframe with averaged sensor values
topsoil_avg = topsoil_sen1_sliced.copy() # create a copy to rewrite
topsoil_avg['T2'] = (topsoil_sen1_sliced['T2'] +
                +   topsoil_sen2_sliced['T2'] +
                +   topsoil_sen3_sliced['T2'])/3
topsoil_avg = topsoil_avg.set_index("DTM")

# --- RESAMPLE TO 10-MINUTE STEP ---
topsoil_avg = topsoil_avg.resample("10min").mean().interpolate()

# Calculate time in seconds from start date and merge with avg temps
time_seconds = (topsoil_avg.index - start_date).total_seconds()

output_data = pd.DataFrame({
    'time[seconds]': time_seconds,
    'temp_0cm[˚C]': topsoil_avg['T2']})

filename = out_FOLD + '102.bc'
with open(filename, 'w') as f:
    f.write(f"# campaign {start_date} {end_date}\n")
    f.write("# time[seconds]\ttemp_0cm[˚C]\n")
    output_data.to_csv(f, sep='\t', index=False, header=False)
