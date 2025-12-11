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
import subprocess
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

    


def runDrutes(par):
    """
    Executes the simulation with a given set of parameters.
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

    # water module
    # organic
    alpha_org = par[6] #  inverse of the air entry suction
    n_org = par[7]  # porosity
    m_org = 1 - 1/n_org
    K_org = par[8] # hydra. conduct.

    # mineral 
    alpha_min = par[9]
    n_min = par[10]
    m_min = 1 - 1/n_min
    K_min = par[11]

    # Build the command to run the shell script.
    cmd = ["bash", "run_drutes_serial.sh",
           str(b1_org),
           str(b2_org),
           str(b3_org),
           str(b1_min),
           str(b2_min),
           str(b3_min),
           str(alpha_org),
           str(n_org),
           str(m_org),
           str(K_org),
           str(alpha_min),
           str(n_min),
           str(m_min),
           str(K_min)
           ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the shell script: {e}")
        return np.inf  # Return a large error if the simulation fails

    error = getError('drutes_run/')
    runDrutes.call_count += 1
    print(f"Iteration: {runDrutes.call_count}\n")

    return error


if __name__ == '__main__':
    # Define bounds for the optimization parameters
    # thermal coef. params
    b1_bnd = (0.02, 1.0) 
    b2_bnd = (0.02, 6.0) 
    b3_bnd = (0.02, 4.0) 
    # van Genuchten params
    K_bnd = (0.000864, 864) # hydro. conduct.
    alpha_bnd = (0.15, 2000) # inverse of air entry suction
    n_bnd = (1.1, 5.0) # porosity
    # Put the into one list
    bounds = [b1_bnd, # organic horizont
              b2_bnd,
              b3_bnd,
              b1_bnd, # mineral horizont
              b2_bnd,
              b3_bnd,
              alpha_bnd, # organic horizont
              n_bnd,
              K_bnd,
              alpha_bnd, # mineral horizont
              n_bnd,
              K_bnd
              ]

    # Initialize a counter attribute for runDrutes
    runDrutes.call_count = 0

    # Run differential evolution optimization in serial
    result = differential_evolution(
        runDrutes,
        bounds,
        strategy='rand1bin',
        popsize=30,
        mutation=(0.5, 1.2),
        recombination=0.8,
        tol=1e-3,
        atol=0,
        maxiter=5000
    )

    # Output the optimized parameter values and error
    print("Optimized values:\n", result.x, '\n', result.fun)
