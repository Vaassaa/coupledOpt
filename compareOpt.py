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

def getData(run_dir):
    """
    Reads the output file from the simulation
    and retrieves the measured data
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

    return measured, simulated

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

    run_dir = f"drutes_run/"

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

    print(f"SIMULATION FINISHED!")


if __name__ == '__main__':

    pars = np.array([
        0.7001064378744838,
        4.007992489039418,
        3.1517895989153364,
        0.6103918429319275,
        5.816931865131173,
        2.0734711171650724,
        1310.6003899411041,
        4.780666751461738,
        89.94921802597821,
        66.92527957567107,
        1.14671967872043,
        715.3293287098194
    ])

    pars = np.array([
        0.6620194587999466,
        5.86750732039099,
        3.0690358297108373,
        0.6232679080018616,
        3.3794070571376054,
        3.3417255804783776,
        1915.3046923452625,
        4.490952362284039,
        298.0170366383968,
        802.6518528004558,
        1.627780169759907,
        445.69436596666105
    ])

    # Run simulation with ^ parameters
    # runDrutes(pars)
    
    # Get simulated and measured values
    measured, simulated = getData("drutes_run/")

    fig, ax = plt.subplots()
    ax.plot(measured['T_8n'])
    ax.plot(simulated['T_8n'])
    fig.savefig("T_8n.png")
    plt.show()
    # column_names = [
            # "time",
            # "T_8n",
            # "T_15n",
            # "T_23n",
            # "theta_8n",
            # "theta_23n"
    # ]
