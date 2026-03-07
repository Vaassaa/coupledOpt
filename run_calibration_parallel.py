"""
Python script for optimalization of parameters
of Saito-Sakai model for modeling of soil temperature
and moisture regime in a forest location of the AMALIA pilot 
intended to run on home workstation
Author: Vaclav Steinbach
Date: 03.01.2026
Dissertation work
"""
import numpy as np
import pandas as pd
import subprocess
from scipy.optimize import differential_evolution
from uuid import uuid4
from datetime import datetime
import shutil
from multiprocessing import Value, Lock
import sys
from calibration_tools import log_run, calcHydraulicHead, shrink_bounds, jitter_init
from functools import partial

# set optimalization strategy
stage = sys.argv[1]

# function call counter for multiprocessing
global_counter = Value('i', 0)    # integer counter
counter_lock   = Lock()

def getError(run_dir):
    """
    Computes a weighted, physics-split RMSE between simulated and measured
    soil temperature and moisture profiles.

    Reads DrUTES observation-point output files and a monitoring CSV from
    `run_dir`, resamples both to a 10-minute grid, and discards the
    pre-warming period (2024-09-08 to 2024-09-12) before evaluating errors.

    Heat residuals are scaled by 1/sigma_T (1 °C) and a signal-intensity
    weight that increases linearly from 1.0 at the minimum recorded
    temperature to 2.0 at the maximum. Moisture residuals are scaled by
    1/sigma_theta (0.05) and a depth weight (2.0 at 8 cm, 1.0 at 23 cm).
    The two component RMSEs are combined in quadrature with equal weights.

    Parameters
    ----------
    run_dir : str
        Path to the DrUTES run directory. Must contain
        `drutes.conf/inverse_modeling/monitoring.dat` and the
        observation-point files under `out/`.

    Returns
    -------
    error : float
        Combined RMSE (quadrature sum of heat and moisture components).
    error_heat : float
        RMSE of the weighted temperature residuals.
    error_moist : float
        RMSE of the weighted soil-moisture residuals.
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
        df_res = df.resample('600s').interpolate()
        # convert timedelta idx back to seconds
        df_res.index = df_res.index.total_seconds().astype(int)
        # keep values only
        series = df_res["theta_l"]

        # reindex to match measured times
        simulated[col] = series.reindex(simulated.index)
    
    # cutoff prewarming time from simulated data
    simulated_cutoff = simulated.query("time > @prewarming_sec")

    # Compute residuals
    diff = simulated_cutoff[column_names[1:]] - measured_cutoff[column_names[1:]]
    # Omit NaNs if present
    diff = diff.dropna()

    # Split by physics
    heat_cols = ["T_8n", "T_15n", "T_23n"]
    moisture_cols = ["theta_8n", "theta_23n"]

    # Calculate a Weight Vector based on the Intensity of the signal
    T_min = measured[heat_cols].min().min()
    T_max = measured[heat_cols].max().max()

    # Peak Weight: 1.0 at the lowest temp, 2.0 at the highest peak
    # weight = 1 + (T - T_min) / (T_max - T_min)
    weights_T = 1.0 + (measured[heat_cols] - T_min) / (T_max - T_min)

    # Calculate a Weight Vector based on the Intensity of the signal
    theta_min = measured[moisture_cols].min().min()
    theta_max = measured[moisture_cols].max().max()

    # Physical normalization scales
    sigma_T     = 1.0    # °C
    sigma_theta = 0.05   # [-]

    # Weighted heat: signal-intensity weight + depth weight
    depth_weights_heat  = {"T_8n": 3.0, "T_15n": 1.5, "T_23n": 1.0}
    depth_weights_moist = {"theta_8n": 2.0, "theta_23n": 1.0}

    # Weighted peaks and depths approach
    for col in heat_cols:
        # diff[col] = (diff[col] / sigma_T) * weights_T[col] * depth_weights_heat[col]
        diff[col] = (diff[col] / sigma_T) * weights_T[col]

    for col in moisture_cols:
        diff[col] = (diff[col] / sigma_theta) * depth_weights_moist[col]

    # Physical normalization approach
    # for col in heat_cols:
        # diff[col] /= sigma_T

    # for col in moisture_cols:
        # diff[col] /= sigma_theta

    # Compute separate errors
    error_heat = np.sqrt(np.mean(diff[heat_cols].values**2))
    error_moist = np.sqrt(np.mean(diff[moisture_cols].values**2))

    # Combine
    # error = error_heat + error_moist
    heat_weight = 1.0
    moist_weight = 1.0
    error = np.sqrt((heat_weight*error_heat)**2 + (moist_weight*error_moist)**2) # quad aggregation
    return error, error_heat, error_moist

    
def runDrutes(strategy, par):
    """
    Executes the simulation with a given set of parameters.
    Parallel execution.
    """
    match strategy:
        case "all":
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
            
            # logaritmic scale
            alpha_org = 10**par[7]
            alpha_min = 10**par[10]
            K_org = 10**par[9]
            K_min = 10**par[12]
            S_max = 10**par[13]

        case "subset":
            # Reconstruct the full parameter set 
            # b1_org = FIXED_PARAMS["b1_org"]
            b1_org = par[0] 
            b2_org = FIXED_PARAMS["b2_org"]
            # b3_org = FIXED_PARAMS["b3_org"]
            b3_org = par[1]
            
            # b1_min = FIXED_PARAMS["b1_min"]
            b1_min = par[2]
            b2_min = FIXED_PARAMS["b2_min"]
            b3_min = FIXED_PARAMS["b3_min"]
            # b3_min = par_subset[2]

            albedo = FIXED_PARAMS["albedo"]
            
            # Water module - Organic
            alpha_org = 10**FIXED_PARAMS["alpha_org"]
            # alpha_org = 10**par_subset[0]
            n_org     = FIXED_PARAMS["n_org"]
            # n_org     = par_subset[1]
            K_org     = 10**FIXED_PARAMS["K_org"]
            # K_org     = 10**par_subset[2]
            m_org     = 1 - 1/n_org
            
            # Water module - Mineral
            # alpha_min = 10**FIXED_PARAMS["alpha_min"]
            alpha_min = 10**par[3]
            # n_min     = FIXED_PARAMS["n_min"]
            n_min     = par[4]
            # K_min     = 10**FIXED_PARAMS["K_min"]
            K_min     = 10**par[5]
            m_min     = 1 - 1/n_min
            
            # Root uptake 
            S_max     = 10**FIXED_PARAMS["S_max"]
            # S_max     = 10**par_subset[6]

    # call counter
    with counter_lock:
        global_counter.value += 1
        call_id = global_counter.value

    # Generate unique dir name for this simulation
    run_id = uuid4().hex
    run_dir = f"drutes_run_{run_id}/"

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

    # fix initial condition
    if h_org < -1e-4 or h_min < -1e-4:
        print(f"SIMULATION {run_id} killed early due to unrealistic hydraulic head:\n"+ 
              +"h_org: {h_org} h_min: {h_min}\n"+
              +"Assigning penalty error.")
        return 1e10  # Return massive error to move optimizer away

    # Build the command to run the shell script.
    cmd = ["bash", "run_drutes.sh", run_dir,
           str(b1_org), str(b2_org), str(b3_org),
           str(b1_min), str(b2_min), str(b3_min),
           str(albedo),
           str(alpha_org), str(n_org), str(m_org), str(K_org),
           str(alpha_min), str(n_min), str(m_min), str(K_min),
           str(S_max),
           str(h_org), str(h_min)]
    
    try:
        #  Adjust timeout based on normal run time.
        print(f"STARTING SIMULATION: {run_id}")
        proc = subprocess.run(cmd, timeout=900, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # proc = subprocess.run(cmd, check=True) # with terminal output (DEBUG)
        
    except subprocess.TimeoutExpired:
        print(f"CRITICAL: Simulation {run_id} TIMED OUT. Assigning penalty error.")
        shutil.rmtree(run_dir, ignore_errors=True)
        return 1e10  # Return massive error to move optimizer away
        
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: Simulation {run_id} CRASHED (Non-convergence).")
        shutil.rmtree(run_dir, ignore_errors=True)
        return 1e10

    # Get the value of objective function
    error, error_heat, error_moist = getError(run_dir)
    print(f"SIMULATION {run_id} FINISHED!")
    print(f"RUN {run_id} OBJECTIVE FUNCTION ERROR: {error}\n")

    # Setup a list of params with correct units
    full_par = [b1_org, b2_org, b3_org,
                b1_min, b2_min, b3_min, 
                albedo,
                alpha_org, n_org, K_org,
                alpha_min, n_min, K_min,
                S_max]
    # Log finished run
    log_run(call_id, error, error_heat, error_moist, full_par_for_log)
    print(f"RUN {run_id} succesfully logged!")

    # Remove temp dir 
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f"RUN {run_id} working dir succesfully removed!")
    return error

# ============================
# --- --- --- MAIN --- --- ---
# ============================
if __name__ == '__main__':
    # Define bounds for the optimization parameters
    # thermal coef. params
    b1_bnd = (0.02, 2.0) 
    b2_bnd = (0.02, 8.0) 
    b3_bnd = (0.02, 6.0) 
    # albedo
    albedo_bnd = (0.05, 0.3) 
    # van Genuchten params
    alpha_bnd = (1, 10) # [1/m] inverse of air entry suction
    n_bnd = (1.05, 7.0) # [-] porosity
    n_org_bnd = (2.25, 5.0) # [-] organic porosity
    n_min_bnd = (1.05, 2.0) # [-] mineral porosity
    K_bnd = (1.0e-6, 3.0e-4) # [m/s] hydro. conduct.

    # van Genuchten params logaritmic
    alpha_bnd = (np.log10(1), np.log10(10)) # [1/m] inverse of air entry suction
    K_bnd = (np.log10(1.0e-8), np.log10(10.0e-4)) # [m/s] hydro. conduct.
    S_max_bnd = (np.log10(1e-9), np.log10(10e-7)) # [m/s] maximum root uptake

    # Put the into one list
    bounds = [b1_bnd, # organic horizont
              b2_bnd,
              b3_bnd,
              b1_bnd, # mineral horizont
              b2_bnd,
              b3_bnd,
              albedo_bnd, # albedo
              alpha_bnd, # organic horizont
              n_min_bnd,
              K_bnd,
              alpha_bnd, # mineral horizont
              n_org_bnd,
              K_bnd,
              S_max_bnd
              ]

    # Subset section first run on sensitivity 
    FIXED_PARAMS = {
        "b2_org": 0.872773,
        "b2_min": 0.854659, 
        "b3_min": 1.4491847,
        "albedo": 0.18,
        "alpha_org": np.log10(4.9273),
        "n_org": 2.09486,
        "K_org": np.log10(5.49161e-5),
        "S_max": np.log10(1.79521e-08)
    }

    # Define bounds for subset
    # thermal coef. params
    b1_org_bnd = (0.02, 3.0) 
    b1_min_bnd = (0.02, 3.0) 
    b3_org_bnd = (0.02, 6.0) 

    # van Genuchten params logaritmic
    alpha_min_bnd = (np.log10(1), np.log10(5)) # [1/m] inverse of air entry suction
    K_min_bnd = (np.log10(1.0e-7), np.log10(10.0e-4)) # [m/s] hydro. conduct.
    n_min_bnd = (1.05, 2.0) # [-] mineral porosity
    bounds_subset = [b1_org_bnd, 
                     b1_min_bnd,
                     b3_org_bnd,
                     alpha_min_bnd,
                     n_min_bnd,
                     K_min_bnd]
    # Defines the log header of what vars were calibrated
    display_subset = ["b1_org", 
                      "b1_min",
                      "b3_org",
                      "alpha_min",
                      "K_min",
                      "n_min"]

    # ========================
    # Optimalization procedure
    # ========================
    match stage:
       # =========================================================
       # Broad search of parameter space with polishing (L-BFGS-B)
       # =========================================================
       case "broad":
           # Define log header for stage 1
           with open("de_log.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMALIZATION LOG --- STAGE 1 --- {timestamp} \n"+
                       "call_id,timestamp,error,error_heat,error_moist,"+
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"+
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"+
                       "albedo[-],"+
                       "alpha_org[1/m],n_org[-],K_org[m/s],"+
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")

           # Run differential evolution optimization in parallel
           result_stage = differential_evolution(
               partial(runDrutes, "all"),
               bounds,
               strategy='rand1bin',
               popsize=16,
               mutation=(0.6, 1.9),
               recombination=0.8,
               tol=1e-3,
               maxiter=100,
               workers=-1,
               updating='deferred',
               polish=True   
           )
       # ==================================================
       # Fine search of parameter space based on best guess
       # ==================================================
       case "fine":
           # Load the best guess array
           best_guess = np.loadtxt("best_guess.in")

           # Shrink the bound around the best case
           # if comming from broad use result_stage1.x as "best_guess"
           refined_bounds = shrink_bounds(best_guess, bounds, shrink=0.15)
           init_pop = jitter_init(best_guess, refined_bounds, rel=0.05, size=16)

           # Define log header for stage 2
           with open("de_log.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMALIZATION LOG --- STAGE 2 --- {timestamp}\n"+
                       "call_id,timestamp,error,error_heat,error_moist,"+
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"+
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"+
                       "albedo[-],"+
                       "alpha_org[1/m],n_org[-],K_org[m/s],"+
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")
           # Run differential evolution optimization in parallel
           result_stage = differential_evolution(
               partial(runDrutes, "all"),
               refined_bounds,
               strategy='best1bin',
               popsize=16,          
               mutation=(0.1, 0.4),
               recombination=0.9,
               tol=1e-5,
               maxiter=300,
               workers=-1,
               updating='deferred',
               polish=True,
               init=init_pop
           )

       # ==================================================
       # Optimize only subset of parameters / rest is fixed    
       # ==================================================
       case "subset":
           # Define log header for subset
           with open("de_log.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMALIZATION LOG --- SUBSET STAGE --- {timestamp}\n"+
                       f"# CALIBRATED VARS: {display_subset}\n"+
                       "call_id,timestamp,error,error_heat,error_moist,"+
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"+
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"+
                       "albedo[-],"+
                       "alpha_org[1/m],n_org[-],K_org[m/s],"+
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")
           # Run differential evolution optimization in parallel
           result_stage = differential_evolution(
               partial(runDrutes, "subset"),
               bounds_subset,
               strategy='rand1bin',
               popsize=16,
               mutation=(0.3, 1.8),
               recombination=0.8,
               tol=1e-3,
               maxiter=80,
               workers=-1,
               updating='deferred',
               polish=True   
           )
       # ====================================================
       # Fine-tune subset parameter space based on best guess
       # ====================================================
       case "subset-fine":
           # Load the best guess array
           best_guess = np.loadtxt("best_guess.in")

           # Shrink the bound around the best case
           # if comming from broad use result_stage1.x as "best_guess"
           refined_bounds = shrink_bounds(best_guess, bounds_subset, shrink=0.15)
           init_pop = jitter_init(best_guess, refined_bounds, rel=0.05, size=16)

           # Define log header for stage 2
           with open("de_log.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMALIZATION LOG --- STAGE 2 --- {timestamp}\n"+
                       "call_id,timestamp,error,error_heat,error_moist,"+
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"+
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"+
                       "albedo[-],"+
                       "alpha_org[1/m],n_org[-],K_org[m/s],"+
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")
           # Run differential evolution optimization in parallel
           result_stage = differential_evolution(
               partial(runDrutes, "subset"),
               refined_bounds,
               strategy='best1bin',
               popsize=16,          
               mutation=(0.1, 0.4),
               recombination=0.9,
               tol=1e-5,
               maxiter=300,
               workers=-1,
               updating='deferred',
               polish=True,
               init=init_pop
           )

    print("CALIBRATION FINISHED!!!")
