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

# function call counter for multiprocessing
global_counter = Value('i', 0)    # integer counter
counter_lock   = Lock()

def shrink_bounds(x, bounds, shrink=0.25):
    new_bounds = []
    for xi, (lo, hi) in zip(x, bounds):
        span = hi - lo
        new_lo = max(lo, xi - shrink * span)
        new_hi = min(hi, xi + shrink * span)
        new_bounds.append((new_lo, new_hi))
    return new_bounds

def log_run(call_id, error, error_heat, error_moist, par, logfile="de_log.txt"):
    """
    Optimalizations results logger
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{call_id}\t"
        f"{timestamp}\t"
        f"{error:.8g}\t"
        f"{error_heat:.8g}\t"
        f"{error_moist:.8g}\t"
        + "\t".join(map(str, par))
        + "\n"
    )
    with open(logfile, "a") as f:
        f.write(line)

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

    # Compute residuals
    diff = simulated[column_names[1:]] - measured[column_names[1:]]
    # Omit NaNs if present
    diff = diff.dropna()

    # Compute normalization constants from measurements
    norm = {}
    for col in diff.columns:
        norm[col] = measured[col].std()

    # Normalize residuals (dimensionless)
    for col in diff.columns:
        diff[col] = diff[col] / norm[col]

    # Split by physics
    heat_cols = ["T_8n", "T_15n", "T_23n"]
    moisture_cols = ["theta_8n", "theta_23n"]

    # Compute separate errors
    error_heat = np.sqrt(np.mean(diff[heat_cols].values**2))
    error_moist = np.sqrt(np.mean(diff[moisture_cols].values**2))

    # Combine
    # error = error_heat + error_moist
    error = np.sqrt(error_heat**2 + error_moist**2) # quad aggregation
    return error, error_heat, error_moist

    
def runDrutes(par):
    """
    Executes the simulation with a given set of parameters.
    Parallel execution.
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

    # call counter
    with counter_lock:
        global_counter.value += 1
        call_id = global_counter.value

    # Generate unique dir name for this simulation
    run_id = uuid4().hex
    run_dir = f"drutes_run_{run_id}/"

    # Build the command to run the shell script.
    cmd = ["bash", "run_drutes_parallel.sh", run_dir,
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

    error, error_heat, error_moist = getError(run_dir)
    print(f"SIMULATION {run_id} FINISHED!")
    print(f"RUN {run_id} OBJECTIVE FUNCTION ERROR: {error}\n")

    # Log finished run
    log_run(call_id, error, error_heat, error_moist, par)

    # Remove temp dir 
    shutil.rmtree(run_dir, ignore_errors=True)
    return error

def jitter_init(x, bounds, rel=0.05, size=16):
    pop = []
    for _ in range(size):
        xi = []
        for v, (lo, hi) in zip(x, bounds):
            span = hi - lo
            dv = np.random.uniform(-rel, rel) * span
            xi.append(np.clip(v + dv, lo, hi))
        pop.append(xi)
    return np.array(pop)


if __name__ == '__main__':
    # Define bounds for the optimization parameters
    # thermal coef. params
    b1_bnd = (0.02, 1.0) 
    b2_bnd = (0.02, 6.0) 
    b3_bnd = (0.02, 4.0) 
    # van Genuchten params
    alpha_bnd = (1, 10) # [1/m] inverse of air entry suction
    n_bnd = (1.05, 3.0) # [-] porosity
    K_bnd = (1.0e-7, 3.0e-4) # [m/s] hydro. conduct.
    # # van Genuchten params
    # alpha_bnd = (0.15, 2000) # inverse of air entry suction
    # n_bnd = (1.1, 5.0) # porosity
    # K_bnd = (0.000864, 864) # hydro. conduct.

    # # Better guess I guess?
    # # thermal coef. params
    # b1_bnd = (0.02, 1.0) 
    # b2_bnd = (3.02, 6.0) 
    # b3_bnd = (1.02, 4.0) 
    # # van Genuchten params
    # alpha_bnd = (1.15, 2000) # inverse of air entry suction
    # n_bnd = (1.1, 5.0) # porosity
    # K_bnd = (0.000864, 864) # hydro. conduct.

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

    # Define log header for stage 1
    with open("de_log.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"OPTIMALIZATION LOG --- STAGE 1 --- {timestamp} \n"+
                "call_id \t timestamp \t error \t error_heat \t error_moist \t"+
                "b1_org \t b2_org \t b3_org \t"+
                "b1_min \t b2_min \t b3_min \t"+
                "alpha_org \t n_org \t K_org \t"+
                "alpha_min \t n_min \t K_min \n")

    # Run differential evolution optimization in parallel
    # First stage run - Aggresive search
    result_stage1 = differential_evolution(
        runDrutes,
        bounds,
        strategy='rand1bin',
        popsize=32,
        mutation=(0.6, 1.2),
        recombination=0.8,
        tol=1e-3,
        maxiter=1,
        workers=-1,
        updating='deferred',
        polish=False   
    )

    # Shrink the bound around the calculated best case
    refined_bounds = shrink_bounds(result_stage1.x, bounds, shrink=0.15)
    init_pop = jitter_init(result_stage1.x, refined_bounds, rel=0.05, size=16)

   # STAGE TWO FROM BEST FOUND 
    # x_best = np.array([
        # 0.7066075248489605,
        # 4.000952981933188,
        # 3.1634828016964063,
        # 0.610669128869678,
        # 5.796028622986176,
        # 2.0409742877617933,
        # 1335.3244091938848,
        # 4.784649120391409,
        # 93.38526453060149,
        # 68.36381671191975,
        # 1.1344025499205599,
        # 702.771330982797
    # ])
    # Shrink the bound around the known best case
    # refined_bounds = shrink_bounds(x_best, bounds, shrink=0.15)
    # init = np.tile(x_best, (8 * len(bounds), 1))
    # init_pop = jitter_init(x_best, refined_bounds, rel=0.05, size=16)

    # Define log header for stage 2
    with open("de_log.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"OPTIMALIZATION LOG --- STAGE 2 --- {timestamp} \n"+
                "call_id \t timestamp \t error \t error_heat \t error_moist \t"+
                "b1_org \t b2_org \t b3_org \t"+
                "b1_min \t b2_min \t b3_min \t"+
                "alpha_org \t n_org \t K_org \t"+
                "alpha_min \t n_min \t K_min \n")

    result_stage2 = differential_evolution(
        runDrutes,
        refined_bounds,
        strategy='best1bin',
        popsize=1,          
        mutation=(0.1, 0.4),
        recombination=0.9,
        tol=1e-5,
        maxiter=300,
        workers=-1,
        updating='deferred',
        polish=True,
        init=init_pop
    )

    print("FINISHED!!!")
