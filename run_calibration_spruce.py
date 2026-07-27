"""
Calibration script for the Spruce stand at AMALIA pilot site.
Calibrates Saito-Sakai model parameters (soil thermal + water) using
Differential Evolution against measured soil temperature and moisture.

Usage:
    python run_calibration_spruce.py <stage>

Stages:
    broad        — full 14-parameter global DE search
    fine         — narrowed DE search from calib_res/best_guess_spruce.in
    subset       — fix thermal/mineral params, calibrate hydraulic subset
    subset-fine  — narrowed subset DE from calib_res/best_guess_newcalib.in
    subset-finer — L-BFGS-B multi-start from best row in de_log_spruce.csv

Author: Vaclav Steinbach
"""
import numpy as np
import pandas as pd
import subprocess
from scipy.optimize import differential_evolution, minimize
from uuid import uuid4
from datetime import datetime
import shutil
from multiprocessing import Value, Lock
import sys
from calibration_tools import log_run, calcHydraulicHead, shrink_bounds, jitter_init
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import os

# Template directory for this stand
DRUTES_TEMPLATE = "drutes_temp_spruce"

# set optimisation strategy from command line
stage = sys.argv[1]

# function call counter for multiprocessing
global_counter = Value('i', 0)
counter_lock   = Lock()


def getError(run_dir):
    """
    Weighted quadrature RMSE between simulated and measured soil
    temperature and moisture profiles.

    Discards the pre-warming period (2024-09-08 to 2024-09-12) before
    evaluating errors. Heat residuals are scaled by 1/sigma_T (1 °C) and
    a signal-intensity weight. Moisture residuals are scaled by
    1/sigma_theta (0.05). Returns (error, error_heat, error_moist).
    """
    column_names = ["time", "T_8n", "T_15n", "T_23n", "theta_8n", "theta_23n"]

    start_warming = datetime(2024, 9, 8, 0, 0, 0)
    end_warming   = datetime(2024, 9, 12, 0, 0, 0)
    prewarming_sec = pd.Timedelta(end_warming - start_warming).total_seconds()

    measured = pd.read_csv(
        run_dir + 'drutes.conf/inverse_modeling/monitoring.dat',
        comment='#', sep='\\s+', header=None, names=column_names, index_col='time'
    )
    measured_cutoff = measured.query("time > @prewarming_sec")

    simulated = measured.copy()

    heat = {"T_8n": "obspt_heat-1.out",
            "T_15n": "obspt_heat-2.out",
            "T_23n": "obspt_heat-3.out"}

    for col, filename in heat.items():
        df = pd.read_csv(
            run_dir + 'out/' + filename,
            comment='#', sep='\\s+', header=None, skiprows=10, engine='python',
            names=['t', 'T', 'flux', 'cum_flux']
        )
        df = df.set_index('t')
        df.index = df.index.astype(int)
        df.index = pd.to_timedelta(df.index, unit="s", errors="coerce")
        df = df[~df.index.duplicated(keep='last')].dropna()
        df_res = df.resample('600s').mean()
        df_res.index = df_res.index.total_seconds().astype(int)
        simulated[col] = df_res["T"].reindex(simulated.index)

    moisture = {"theta_8n": "obspt_RE_matrix-1.out",
                "theta_23n": "obspt_RE_matrix-3.out"}

    for col, filename in moisture.items():
        df = pd.read_csv(
            run_dir + 'out/' + filename,
            comment='#', sep='\\s+', header=None, skiprows=10, engine='python',
            names=['t', 'h', 'theta_l', 'theta_v', 'l_flux', 'cum_flux',
                   'v_flux', 'cum_l_flux', 'tot_flux', 'total_flux', 'cum_flux2']
        )
        df = df.set_index('t')
        df.index = df.index.astype(float).round().astype(int)
        df.index = pd.to_timedelta(df.index, unit="s", errors="coerce")
        df = df[~df.index.duplicated(keep='last')].dropna()
        df_res = df.resample('600s').mean()
        df_res.index = df_res.index.total_seconds().astype(int)
        simulated[col] = df_res["theta_l"].reindex(simulated.index)

    simulated_cutoff = simulated.query("time > @prewarming_sec")

    diff = (simulated_cutoff[column_names[1:]] - measured_cutoff[column_names[1:]]).dropna()

    heat_cols     = ["T_8n", "T_15n", "T_23n"]
    moisture_cols = ["theta_8n", "theta_23n"]

    T_mean  = measured[heat_cols].mean().mean()
    T_range = (measured[heat_cols].max().max() - measured[heat_cols].min().min()) / 2
    weights_T = 1.0 + (measured[heat_cols] - T_mean).abs() / T_range

    sigma_T     = 1.0
    sigma_theta = 0.05

    for col in heat_cols:
        diff[col] = (diff[col] / sigma_T) * weights_T[col]
    for col in moisture_cols:
        diff[col] = diff[col] / sigma_theta

    error_heat  = np.sqrt(np.mean(diff[heat_cols].values**2))
    error_moist = np.sqrt(np.mean(diff[moisture_cols].values**2))
    error = np.sqrt(error_heat**2 + error_moist**2)
    return error, error_heat, error_moist


def runDrutes(strategy, log, par):
    """
    Run a single DRUtES simulation and return the objective function error.

    Unpacks par according to strategy:
      "all"    — 14 parameters (full optimisation)
      "subset" — 9 parameters; remaining drawn from FIXED_PARAMS

    Rejects parameter combinations that produce h_org or h_min < -500 m
    (drier than deep wilting point) with a penalty of 1e10.
    """
    match strategy:
        case "all":
            b1_org = par[0]
            b2_org = par[1]
            b3_org = par[2]
            b1_min = par[3]
            b2_min = par[4]
            b3_min = par[5]
            albedo = par[6]

            alpha_org = 10**par[7]
            n_org     = par[8]
            m_org     = 1 - 1/n_org
            K_org     = 10**par[9]

            alpha_min = 10**par[10]
            n_min     = par[11]
            m_min     = 1 - 1/n_min
            K_min     = 10**par[12]

            S_max     = 10**par[13]

        case "subset":
            b1_org = par[0]
            b2_org = FIXED_PARAMS["b2_org"]
            b3_org = par[1]

            b1_min = FIXED_PARAMS["b1_min"]
            b2_min = FIXED_PARAMS["b2_min"]
            b3_min = FIXED_PARAMS["b3_min"]

            albedo = par[2]

            alpha_org = 10**par[3]
            n_org     = par[4]
            K_org     = 10**par[5]
            m_org     = 1 - 1/n_org

            alpha_min = 10**par[6]
            n_min     = FIXED_PARAMS["n_min"]
            K_min     = 10**par[7]
            m_min     = 1 - 1/n_min

            S_max     = 10**par[8]

    with counter_lock:
        global_counter.value += 1
        call_id = global_counter.value

    run_id  = uuid4().hex
    run_dir = f"drutes_run_{run_id}/"

    monitoring = pd.read_csv(
        DRUTES_TEMPLATE + '/drutes.conf/inverse_modeling/monitoring.dat',
        comment='#', sep='\\s+', header=None
    )
    theta_org = monitoring.iloc[0, 4]
    theta_min = monitoring.iloc[0, 5]
    h_org = calcHydraulicHead(theta_org, [alpha_org, n_org, 1 - 1/n_org])
    h_min = calcHydraulicHead(theta_min, [alpha_min, n_min, 1 - 1/n_min])

    if h_org < -500 or h_min < -500:
        print(f"SIMULATION {run_id} killed early — unrealistic head: "
              f"h_org={h_org:.1f}  h_min={h_min:.1f}")
        return 1e10

    cmd = ["bash", "run_drutes.sh", DRUTES_TEMPLATE, run_dir,
           str(b1_org), str(b2_org), str(b3_org),
           str(b1_min), str(b2_min), str(b3_min),
           str(albedo),
           str(alpha_org), str(n_org), str(1 - 1/n_org), str(K_org),
           str(alpha_min), str(n_min), str(1 - 1/n_min), str(K_min),
           str(S_max),
           str(h_org), str(h_min)]

    try:
        print(f"STARTING SIMULATION: {run_id}")
        subprocess.run(cmd, timeout=900, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        error, error_heat, error_moist = getError(run_dir)
        print(f"SIMULATION {run_id} FINISHED! error={error:.6f}")

        full_par = [b1_org, b2_org, b3_org,
                    b1_min, b2_min, b3_min,
                    albedo,
                    alpha_org, n_org, K_org,
                    alpha_min, n_min, K_min,
                    S_max]

        logfile = "finetune_log_spruce.csv" if log == "fine" else "de_log_spruce.csv"
        log_run(call_id, error, error_heat, error_moist, full_par, logfile)
        print(f"RUN {run_id} logged!")

    except subprocess.TimeoutExpired:
        print(f"CRITICAL: Simulation {run_id} TIMED OUT.")
        os.makedirs("nonconvergent/", exist_ok=True)
        dest = os.path.join("nonconvergent/", os.path.basename(run_dir))
        shutil.move(run_dir, dest)
        shutil.rmtree(os.path.join(dest, "bin/"), ignore_errors=True)
        shutil.rmtree(os.path.join(dest, "out/"), ignore_errors=True)
        return 1e10

    except subprocess.CalledProcessError:
        print(f"CRITICAL: Simulation {run_id} CRASHED.")
        shutil.rmtree(run_dir, ignore_errors=True)
        return 1e10

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"RUN {run_id} working dir removed!")

    return error


# ============================
# --- --- --- MAIN --- --- ---
# ============================
if __name__ == '__main__':

    # ------------------------------------------------------------------ #
    # Shared bounds for "broad" and "fine" stages (all 14 parameters)     #
    # ------------------------------------------------------------------ #
    b1_bnd     = (0.02, 2.0)
    b2_bnd     = (0.02, 8.0)
    b3_bnd     = (0.02, 6.0)
    albedo_bnd = (0.05, 1.0)

    alpha_bnd   = (np.log10(1), np.log10(10))
    n_org_bnd   = (2.25, 5.0)
    n_min_bnd   = (1.05, 2.0)
    K_bnd       = (np.log10(1.0e-8), np.log10(10.0e-4))
    S_max_bnd   = (np.log10(1e-9), np.log10(10e-6))

    bounds = [b1_bnd,      # par[0]  b1_org
              b2_bnd,      # par[1]  b2_org
              b3_bnd,      # par[2]  b3_org
              b1_bnd,      # par[3]  b1_min
              b2_bnd,      # par[4]  b2_min
              b3_bnd,      # par[5]  b3_min
              albedo_bnd,  # par[6]  albedo
              alpha_bnd,   # par[7]  alpha_org  (log10)
              n_org_bnd,   # par[8]  n_org
              K_bnd,       # par[9]  K_org      (log10)
              alpha_bnd,   # par[10] alpha_min  (log10)
              n_min_bnd,   # par[11] n_min
              K_bnd,       # par[12] K_min      (log10)
              S_max_bnd,   # par[13] S_max      (log10)
              ]

    # ------------------------------------------------------------------ #
    # SPRUCE subset configuration                                          #
    # Update FIXED_PARAMS with best "all"-stage values before running     #
    # "subset". n_min=2.875 was obtained from a previous spruce run.      #
    # ------------------------------------------------------------------ #
    FIXED_PARAMS = {
        "b2_org": 6.4152019,
        "b1_min": 0.22083929,
        "b2_min": 0.51442833,
        "b3_min": 0.51171692,
        "n_min":  2.8750303,
    }

    b1_org_bnd    = (0.02, 10.0)
    b3_org_bnd    = (0.02, 10.0)
    albedo_bnd    = (0.05, 1.0)
    alpha_org_bnd = (np.log10(1), np.log10(7))
    n_org_bnd_sub = (1.05, 6.0)
    K_org_bnd     = (np.log10(1.0e-8), np.log10(10.0e-4))
    alpha_min_bnd = (np.log10(1), np.log10(7))
    K_min_bnd     = (np.log10(1.0e-8), np.log10(10.0e-4))
    S_max_bnd_sub = (np.log10(1e-9), np.log10(10e-6))

    # 9 elements matching the "subset" case in runDrutes:
    # par[0]=b1_org, [1]=b3_org, [2]=albedo, [3]=alpha_org, [4]=n_org,
    # [5]=K_org, [6]=alpha_min, [7]=K_min, [8]=S_max
    bounds_subset = [b1_org_bnd,
                     b3_org_bnd,
                     albedo_bnd,
                     alpha_org_bnd,
                     n_org_bnd_sub,
                     K_org_bnd,
                     alpha_min_bnd,
                     K_min_bnd,
                     S_max_bnd_sub]

    display_subset = ["b1_org", "b3_org", "albedo",
                      "alpha_org", "n_org", "K_org",
                      "alpha_min", "K_min", "S_max"]

    # ================================================================== #
    # Optimisation stages                                                  #
    # ================================================================== #
    match stage:

       # ---------------------------------------------------------------- #
       # Broad: global search over all 14 parameters                      #
       # ---------------------------------------------------------------- #
       case "broad":
           with open("de_log_spruce.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMISATION LOG --- SPRUCE BROAD --- {timestamp}\n"
                       "call_id,timestamp,error,error_heat,error_moist,"
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"
                       "albedo[-],"
                       "alpha_org[1/m],n_org[-],K_org[m/s],"
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")

           result_stage = differential_evolution(
               partial(runDrutes, "all", "broad"),
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

       # ---------------------------------------------------------------- #
       # Fine: narrowed search seeded from best_guess_spruce.in           #
       # ---------------------------------------------------------------- #
       case "fine":
           best_guess = np.loadtxt("calib_res/best_guess_spruce.in")

           best_guess_log = best_guess.copy()
           best_guess_log[7]  = np.log10(best_guess[7])   # alpha_org
           best_guess_log[9]  = np.log10(best_guess[9])   # K_org
           best_guess_log[10] = np.log10(best_guess[10])  # alpha_min
           best_guess_log[12] = np.log10(best_guess[12])  # K_min
           best_guess_log[13] = np.log10(best_guess[13])  # S_max

           refined_bounds = shrink_bounds(best_guess_log, bounds, shrink=0.15)
           init_pop       = jitter_init(best_guess_log, refined_bounds, rel=0.05, size=16)

           with open("finetune_log_spruce.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMISATION LOG --- SPRUCE FINE --- {timestamp}\n"
                       "call_id,timestamp,error,error_heat,error_moist,"
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"
                       "albedo[-],"
                       "alpha_org[1/m],n_org[-],K_org[m/s],"
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")

           result_stage = differential_evolution(
               partial(runDrutes, "all", "fine"),
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

       # ---------------------------------------------------------------- #
       # Subset: optimise 9 hydraulic+thermal params; rest fixed          #
       # Update FIXED_PARAMS above before running this stage.             #
       # ---------------------------------------------------------------- #
       case "subset":
           with open("de_log_spruce.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMISATION LOG --- SPRUCE SUBSET --- {timestamp}\n"
                       f"# CALIBRATED VARS: {display_subset}\n"
                       "call_id,timestamp,error,error_heat,error_moist,"
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"
                       "albedo[-],"
                       "alpha_org[1/m],n_org[-],K_org[m/s],"
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")

           result_stage = differential_evolution(
               partial(runDrutes, "subset", "broad"),
               bounds_subset,
               strategy='rand1bin',
               popsize=16,
               mutation=(0.3, 1.8),
               recombination=0.8,
               tol=1e-3,
               maxiter=42,
               workers=-1,
               updating='deferred',
               polish=True
           )

       # ---------------------------------------------------------------- #
       # Subset-fine: narrowed subset DE from best_guess_newcalib.in      #
       # ---------------------------------------------------------------- #
       case "subset-fine":
           best_guess = np.loadtxt("calib_res/best_guess_newcalib.in")

           best_guess_log = best_guess.copy()
           best_guess_log[7]  = np.log10(best_guess[7])   # alpha_org
           best_guess_log[9]  = np.log10(best_guess[9])   # K_org
           best_guess_log[10] = np.log10(best_guess[10])  # alpha_min
           best_guess_log[12] = np.log10(best_guess[12])  # K_min
           best_guess_log[13] = np.log10(best_guess[13])  # S_max

           refined_bounds = shrink_bounds(best_guess_log, bounds_subset, shrink=0.15)
           init_pop       = jitter_init(best_guess_log, refined_bounds, rel=0.05, size=16)

           with open("finetune_log_spruce.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMISATION LOG --- SPRUCE SUBSET-FINE --- {timestamp}\n"
                       "call_id,timestamp,error,error_heat,error_moist,"
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"
                       "albedo[-],"
                       "alpha_org[1/m],n_org[-],K_org[m/s],"
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")

           result_stage = differential_evolution(
               partial(runDrutes, "subset", "fine"),
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

       # ---------------------------------------------------------------- #
       # Subset-finer: L-BFGS-B multi-start from best row in log         #
       # ---------------------------------------------------------------- #
       case "subset-finer":
           log_df = pd.read_csv("de_log_spruce.csv", comment="#", header=0)
           min_error_idx = log_df["error"].idxmin()
           min_error_val = log_df["error"][min_error_idx]
           print(f"Best guess found in log: {min_error_val:.6f} (row {min_error_idx + 4})")
           best_row = log_df.loc[min_error_idx]

           calibrated_cols = ["b1_org[W/(m.K)]", "b3_org[W/(m.K)]",
                              "albedo[-]",
                              "alpha_org[1/m]", "n_org[-]", "K_org[m/s]",
                              "alpha_min[1/m]", "K_min[m/s]", "S_max[m/s]"]

           best_guess = best_row[calibrated_cols].to_numpy(dtype=float)
           print(f"best_guess shape: {best_guess.shape}, bounds_subset length: {len(bounds_subset)}")
           print(best_guess)

           best_guess[3] = np.log10(best_guess[3])   # alpha_org
           best_guess[5] = np.log10(best_guess[5])   # K_org
           best_guess[6] = np.log10(best_guess[6])   # alpha_min
           best_guess[7] = np.log10(best_guess[7])   # K_min
           best_guess[8] = np.log10(best_guess[8])   # S_max

           rng      = np.random.default_rng(42)
           init_pop = best_guess + rng.uniform(-0.01, 0.01, size=(16, len(best_guess))) * best_guess

           with open("finetune_log_spruce.csv", "a") as f:
               timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               f.write(f"\n# OPTIMISATION LOG --- SPRUCE SUBSET-FINER (L-BFGS-B) --- {timestamp}\n"
                       f"# CALIBRATED VARS: {display_subset}\n"
                       "call_id,timestamp,error,error_heat,error_moist,"
                       "b1_org[W/(m.K)],b2_org[W/(m.K)],b3_org[W/(m.K)],"
                       "b1_min[W/(m.K)],b2_min[W/(m.K)],b3_min[W/(m.K)],"
                       "albedo[-],"
                       "alpha_org[1/m],n_org[-],K_org[m/s],"
                       "alpha_min[1/m],n_min[-],K_min[m/s],S_max[m/s]\n")

           obj_fn = partial(runDrutes, "subset", "fine")

           def run_single(x0):
               return minimize(
                   obj_fn, x0,
                   method="L-BFGS-B",
                   bounds=bounds_subset,
                   options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8}
               )

           with ProcessPoolExecutor() as executor:
               results = list(executor.map(run_single, init_pop))

           result_stage = min(results, key=lambda r: r.fun if r.success else np.inf)

    print("!!! CALIBRATION FINISHED !!!")
