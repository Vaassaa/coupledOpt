"""
Python script that computes daily precipitation
based on drutes simulation input data
Author: Vaclav Steinbach
Date: 07.01.2026
Dissertation
"""
import pandas as pd
from scipy import integrate

# --- CONFIG --- 
filename = 'rain.in'

# Define column names according to the commented header.
column_names = [
    "time[s]", 
    "precipitation[m/s]"
]
# Load precipitation data
data = pd.read_csv(filename, comment='#', sep='\\s+', header=None, names=column_names)

# Check data
print(data.info())
print("---------------------")

# Get last entry
last_entry = data['time[s]'].iloc[-1]
# Sum all data in the column
total_sum = data['precipitation[m/s]'].sum()
# Get number of days
days = last_entry/(24*3600) # seconds -> days
print(f"Number of days: {days}")
# Get timestep
timestep = data['time[s]'].iloc[1] - data['time[s]'].iloc[0]
# Calculate total daily precipitation (integral)
total_precip = total_sum*timestep/days
print(f"Daily precipitation: {total_precip} meters\n")
