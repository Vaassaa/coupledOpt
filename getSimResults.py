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
from datetime import datetime

run = sys.argv[1]

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
            "T_23n": "obspt_heat-3.out"}

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
    moisture = {"theta_8n": "obspt_RE_matrix-1.out",
                "theta_23n": "obspt_RE_matrix-3.out"}

    # run through simulated moisture data and assign them into dataframe
    for col, filename in moisture.items():
        df = pd.read_csv(run_dir+'out/'+filename,
                                comment = '#', sep='\\s+', header = None, skiprows=10, engine='python',
                                names = ['t',
                                         'h',
                                         'theta_l',
                                         'theta_v',
                                         'l_flux',
                                         'cum_flux',
                                         'v_flux',
                                         'cum_l_flux',
                                         'tot_flux',
                                         'total_flux',
                                         'cum_flux2'])

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
        series = df_res["theta_l"]

        # reindex to match measured times
        simulated[col] = series.reindex(simulated.index)

    # cutoff prewarming time from simulated data
    simulated_cutoff = simulated.query("time > @prewarming_sec")

    return measured_cutoff, simulated_cutoff

def runDrutes(par):
    """
    Runs a single DRUtES simulation with a fixed 14-parameter vector.

    Unpacks `par` into thermal coefficients (b1–b3), albedo, and van
    Genuchten hydraulic parameters (alpha, n, K) for organic and mineral
    horizons, plus maximum root uptake S_max. 

    Initial hydraulic heads are computed from the first moisture readings
    in the monitoring file and passed to the simulation as initial
    conditions. All parameters are forwarded to `run_drutes.sh` via
    subprocess; a non-zero exit code raises CalledProcessError.

    Parameters
    ----------
    par : array-like of float, length 14
        [b1_org, b2_org, b3_org, b1_min, b2_min, b3_min, albedo,
         alpha_org, n_org, K_org, alpha_min, n_min, K_min, S_max]

    Returns
    -------
    None
    """ 
    # Define input parameters   
    # evap module
    # organic
    b1_org = par[0] # thermal coef. pars
    b2_org = par[1]
    b3_org = par[2]

    # mineral
    b1_min = par[3]
    b2_min = par[4]
    b3_min = par[5]

    # albedo
    albedo = par[6]

    # water module
    # organic
    alpha_org = par[7] #  inverse of the air entry suction
    n_org = par[8]  # porosity
    m_org = 1 - 1/n_org
    K_org = par[9] # hydra. conduct. logaritmic scale

    # mineral 
    alpha_min = par[10] # logaritmic scale
    n_min = par[11]
    m_min = 1 - 1/n_min
    K_min = par[12] # logaritmic scale
    S_max = par[13] # maximum root uptake

    run_dir = f"drutes_run/"

    # compute hydraulic head for intial theta
    # early stopping of simulation from unrealistic params
    monitoring = pd.read_csv('drutes_temp/drutes.conf/inverse_modeling/monitoring.dat',
                           comment = '#', sep='\\s+', header = None)
    # get initial soil moisture
    theta_org = monitoring.iloc[0,4] 
    theta_min = monitoring.iloc[0,5] 
    # compute hydraulic head for each horizont
    h_org = calcHydraulicHead(theta_org, [alpha_org, n_org, m_org])
    h_min = calcHydraulicHead(theta_min, [alpha_min, n_min, m_min])

    # Build the command to run t e shell script.
    cmd = ["bash", "run_drutes.sh", run_dir,
           str(b1_org), str(b2_org), str(b3_org),
           str(b1_min), str(b2_min), str(b3_min),
           str(albedo),
           str(alpha_org), str(n_org), str(m_org), str(K_org),
           str(alpha_min), str(n_min), str(m_min), str(K_min),
           str(S_max),
           str(h_org), str(h_min)]
    
    # run the command
    subprocess.run(cmd, check=True)
    print(f"SIMULATION FINISHED!")


if __name__ == '__main__':
    pars = np.loadtxt('calib_res/beech_params_tighter_formatted.in')
    print(pars)

    # # Run simulation with ^ parameters
    match run:
        case "calc":
            runDrutes(pars)
        case "plot":
            # Get simulated and measured values
            measured, simulated = getData("drutes_run/")
            var_names = [
                    "time", # 0
                    "T_8n", # 1
                    "T_15n", # 2
                    "T_23n", # 3
                    "theta_8n", # 4
                    "theta_23n" # 5
            ]
            # Select a varible to plot
            var = var_names[1]

            # if var == 

            fig, ax = plt.subplots()
            ax.plot(measured[var], label = "measured")
            ax.plot(simulated[var], label = "simulated")
            # ax.set_ylim([0,0.25])
            ax.set_ylim([0,0.5225]) # theta_s organic
            ax.set_ylim([0,0.4200]) # theta_s mineral
            ax.legend()
            ax.grid()
            fig.savefig("figs/best_"+var)
            plt.show()


