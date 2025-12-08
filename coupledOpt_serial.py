"""
Python script for optimalization of parameters
of Saito-Sakai model for modeling of soil temperature
and moisture regime in a forest location of the AMALIA pilot 
intended to run on metacentrum cluster
Author: Vaclav Steinbach
Date: 08.12.2025
Dissertation work
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

def getError(run_dir):
    """
    Reads the output file from the simulation 
    and computes the error as the root-sum-of-squares 
    of the simulated monitoring values.
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

    # load monitoring data
    measured = pd.read_csv(run_dir+'drutes.conf/inverse_modeling/monitoring.dat',
                           comment = '#', sep='\\s+', header = None, names = column_names, index_col='time')
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
        # drop NaNs
        df = df.dropna()
        # resample to 10 min
        df_res = df.resample('600s').mean()
        # convert timedelta idx back to seconds
        df_res.index = df_res.index.total_seconds().astype(int)
        # keep values only
        series = df_res["T"] / 100 # for better comparability in RMSE

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

    # Compute the RMSE
    diff = simulated[column_names[1:]] - measured[column_names[1:]]
    error = np.sqrt(np.sum(diff.values**2))
    print(simulated.head())
    print(measured.head())
    return error

    


# def runDrutes(par):
    # """
    # Executes the simulation with a given set of parameters.
    # A unique temporary working directory is created for each run.
    # """
    
    # return error

if __name__ == '__main__':
    getError('drutes_run/')
    
