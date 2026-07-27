"""
Python script for comparison of simulation output of Saito-Sakai model 
for of soil temperature and moisture regime in a forest location
of the AMALIA pilot and measured data
Author: Vaclav Steinbach
Date: 05.01.2026
Dissertation work
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from calibration_tools import calcHydraulicHead
import sys
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy import integrate


def getData(run_dir):
    """
    Loads and aligns measured and simulated soil temperature and moisture
    profiles onto a common 10-minute time grid.

    Reads the monitoring CSV and DRUtES observation-point output files from
    `run_dir`. Simulated heat (3 depths) and liquid moisture (2 depths) are
    each resampled to 600 s intervals and reindexed to match the measured
    time axis. Duplicate timestamps are dropped before resampling.
    The pre-warming period is stripped from the returned DataFrames

    Parameters
    ----------
    run_dir : str
        Path to the DrUTES run directory. Must contain
        `drutes.conf/inverse_modeling/monitoring.dat` and observation-point
        files under `out/`.

    Returns
    -------
    measured : pd.DataFrame
        Monitoring data with columns [T_8n, T_15n, T_23n, theta_8n, theta_23n],
        indexed by time in seconds.
    simulated : pd.DataFrame
        Simulated counterpart on the same index, populated from DrUTES
        observation-point output files.
    """
    # define column names for both measured and simulated dataframe
    column_names = [
            "time",
            "T_8n",
            "T_15n",
            "T_23n",
            "theta_8n",
            "theta_23n"
    ]
    # select date for prewarming of model
    start_warming = datetime(2024, 9, 8, 0, 0, 0)
    end_warming = datetime(2024, 9, 12, 0, 0, 0)

    # compute the number of seconds between the two dates
    prewarming = end_warming - start_warming
    prewarming_sec = pd.Timedelta(prewarming).total_seconds()

    # load monitoring data
    measured = pd.read_csv(run_dir+'drutes.conf/inverse_modeling/monitoring.dat',
                           comment = '#', sep='\\s+', header = None, names = column_names, index_col='time')
    # cutoff prewarming time from measured data
    measured_cutoff = measured.query("time > @prewarming_sec")
    # create a new dataframe of the same size
    simulated = measured.copy()

    # define the filenames of observed points for theta
    heat = {"T_8n": "obspt_heat-1.out",
            "T_15n": "obspt_heat-2.out",
            "T_23n": "obspt_heat-3.out",
            "T_125n": "obspt_heat-4.out"}

    # run through simulated heat data and assign them into dataframe
    for col, filename in heat.items():
        df = pd.read_csv(run_dir+'out/'+filename,
                                comment = '#', sep='\\s+', header = None, skiprows=10, engine='python',
                                names = ['t','T','flux','cum_flux'])

        # set index to time for resampling
        df = df.set_index('t')
        # set index from float to int
        df.index = df.index.astype(int)
        # convert s -> datetime (for resampling)
        df.index = pd.to_timedelta(df.index, unit="s", errors="coerce")
        # drop duplicates
        df = df[~df.index.duplicated(keep='last')]
        # drop NaNs
        df = df.dropna()
        # resample to 10 min
        df_res = df.resample('600s').mean()
        # convert timedelta idx back to seconds
        df_res.index = df_res.index.total_seconds().astype(int)
        # keep values only
        series = df_res["T"] # for better comparability in RMSE

        # reindex to match measured times
        simulated[col] = series.reindex(simulated.index)
        
    
    # define the filenames of observed points for temperature
    moisture = {"l_flux_8n": "obspt_RE_matrix-1.out",
                "l_flux_23n": "obspt_RE_matrix-3.out",
                "l_flux_125n": "obspt_RE_matrix-4.out"}

    # run through simulated moisture data and assign them into dataframe
    for col, filename in moisture.items():
        df = pd.read_csv(run_dir+'out/'+filename,
                                comment = '#', sep='\\s+', header = None, skiprows=10, engine='python',
                                names = ['t',
                                         'h',
                                         'theta_l',
                                         'theta_v',
                                         'sat_degree',
                                         'l_flux',
                                         'cum_l_flux',
                                         'v_flux',
                                         'cum_v_flux',
                                         'total_flux',
                                         'cum_total_flux'])

        # set index to time for resampling
        df = df.set_index('t')
        # set index from float to int
        df.index = df.index.astype(float).round().astype(int)
        # convert s -> datetime (for resampling)
        df.index = pd.to_timedelta(df.index, unit="s", errors="coerce")
        # drop duplicates
        df = df[~df.index.duplicated(keep='last')]
        # drop NaNs
        df = df.dropna()
        # resample to 10 min
        df_res = df.resample('600s').mean()
        # convert timedelta idx back to seconds
        df_res.index = df_res.index.total_seconds().astype(int)
        # keep values only
        series = df_res["l_flux"]

        # reindex to match measured times
        simulated[col] = series.reindex(simulated.index)

    # cutoff prewarming time from simulated data
    simulated_cutoff = simulated.query("time > @prewarming_sec")

    return measured_cutoff, simulated_cutoff

if __name__ == '__main__':
    # ==============
    # --- CONFIG --- 
    # ==============
    # Get simulated and measured values
    measured_beech, simulated_beech = getData("arch/beech/drutes_run/")
    measured_spruce, simulated_spruce = getData("arch/spruce/drutes_run/")

    # Calculate integral from both curves
    beech_retention = integrate.cumulative_trapezoid(simulated_beech["l_flux_23n"], measured_beech.index, initial=0)
    spruce_retention = integrate.cumulative_trapezoid(simulated_spruce["l_flux_23n"], measured_beech.index, initial=0)
    print(f"RETENTION SPRUCE {spruce_retention[-1]} / BEECH {beech_retention[-1]}")
    var_names = [
            "time", # 0
            "l_flux_8n", # 1
            "l_flux_23n", # 2
            "l_flux_125n", # 3
    ]

    # Convert time (seconds) to datetime
    start_dt = datetime(2024, 9, 8, 0, 0, 0)
    datetime_index = [start_dt + timedelta(seconds=int(t)) for t in measured_beech.index]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(datetime_index, simulated_beech[var_names[3]], color='royalblue', label='beech')
    ax.plot(datetime_index, simulated_spruce[var_names[3]], color='darkblue', label='spruce')
    ax.set_ylabel(r"$liquid flux_{-125 cm}$ [m/s]")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.42)
    # --- Shared x-axis formatting ---
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=12, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_xlim([datetime_index[0], datetime_index[-1]])

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig("figs/trees_retention_deeper.png", dpi=300)
    plt.show()
