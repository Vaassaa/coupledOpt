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
        0.5099648456240723,
        4.866545105179517,
        2.0155925462908812,
        0.9030764750219564,
        1.8760483434023754,
        2.522422816117581,
        9.906938396060422,
        1.4217497650423456,
        9.720456055924286e-05,
        9.311262631837844,
        1.267783779784333,
        0.0001571977257848784
    ])

# b1_org 	 b2_org 	 b3_org 	b1_min 	 b2_min 	 b3_min 	alpha_org 	 n_org 	 K_org 	alpha_min 	 n_min 	 K_min 
    pars = np.array([
        0.04871360318978818,
        1.9256419514950864,
        0.4693011731841115,	
        0.8449347494088193,
        0.7772501389644668,
        2.517285480004568,
        10**(0.14233203226881186),
        2.10011125096081,
        10**(-5.329672031909213),
        10**(0.8931143202528847),
        1.3023123355105821,
        10**(-4.1656430577378964)
    ])

    pars = np.array([
        0.5099648456240723,
        4.866545105179517,
        2.0155925462908812,
        0.9030764750219564,
        1.8760483434023754,
        2.522422816117581,
        10**(0.14233203226881186),
        2.10011125096081,
        10**(-5.329672031909213),
        10**(0.8931143202528847),
        1.3023123355105821,
        10**(-4.1656430577378964)
    ])

    # Best Params T +- 2C theta +- 0.05
    pars = np.array([
	0.3023971185123264,
    1.590335481782116,
    0.28565403481747753,
    0.7710899131548179,
    3.2657499091555615,
    2.427112295530444,
    10**(0.03620358478194792),
    1.9509796999625308,
    10**(-3.5829128049564236),
    10**(0.11108108129699978),
    1.4982511125160891,
    10**(-4.333430501103598)
    ])

    pars = np.array([
	0.23641580527590275,
    0.7870152960811867,
    0.06598357592989612,
    0.7035906332191479,
    5.611625300170809,
    3.143169577875028,
    10**(0.4894974285141164),
    2.121011576780523,
    10**(-4.1923211861588126),
    10**(0.19588417770225458),
    1.5769741336863135,
    10**(-4.068826530722949)
    ])
    # Run simulation with ^ parameters
    runDrutes(pars)
    
    # Get simulated and measured values
    measured, simulated = getData("drutes_run/")

    fig, ax = plt.subplots()
    var = "theta_8n"
    ax.plot(measured[var])
    ax.plot(simulated[var])
    fig.savefig("figs/best_"+var)
    plt.show()
    # column_names = [
            # "time",
            # "T_8n",
            # "T_15n",
            # "T_23n",
            # "theta_8n",
            # "theta_23n"
    # ]
