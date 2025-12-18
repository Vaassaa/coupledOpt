"""
Python script for optimization of parameters
of Saito-Sakai model for modeling of soil temperature
and moisture regime in a forest location of the AMALIA pilot
intended to run on metacentrum cluster

Author: Vaclav Steinbach
Date: 08.12.2025
Dissertation work
"""

import os
import numpy as np
import pandas as pd
import subprocess
from scipy.optimize import differential_evolution
from uuid import uuid4
from datetime import datetime
import shutil
from multiprocessing import Value, Lock

# ==============================
# GLOBAL CONFIG
# ==============================

FAIL_PENALTY = 1e12  # finite penalty for failed runs

# Prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# function call counter for multiprocessing
global_counter = Value('i', 0)
counter_lock = Lock()


# ==============================
# LOGGING
# ==============================

def log_run(call_id, error, par, logfile="de_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{call_id}\t"
        f"{timestamp}\t"
        f"{error:.8g}\t"
        + "\t".join(map(str, par))
        + "\n"
    )
    with open(logfile, "a") as f:
        f.write(line)


# ==============================
# ERROR EVALUATION
# ==============================

def getError(run_dir):
    column_names = [
        "time",
        "T_8n",
        "T_15n",
        "T_23n",
        "theta_8n",
        "theta_23n"
    ]

    monitoring_file = os.path.join(
        run_dir, "drutes.conf/inverse_modeling/monitoring.dat"
    )

    if not os.path.exists(monitoring_file):
        raise FileNotFoundError("Monitoring data missing")

    measured = pd.read_csv(
        monitoring_file,
        comment='#',
        sep=r'\s+',
        header=None,
        names=column_names,
        index_col='time'
    )

    simulated = measured.copy()

    # ---------- Heat ----------
    heat = {
        "T_8n": "obspt_heat-1.out",
        "T_15n": "obspt_heat-2.out",
        "T_23n": "obspt_heat-3.out"
    }

    for col, filename in heat.items():
        fpath = os.path.join(run_dir, "out", filename)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {filename}")

        df = pd.read_csv(
            fpath,
            comment='#',
            sep=r'\s+',
            header=None,
            skiprows=10,
            engine='python',
            names=['t', 'T', 'flux', 'cum_flux']
        )

        df = df.set_index('t')
        df.index = df.index.astype(float).round().astype(int)
        df.index = pd.to_timedelta(df.index, unit="s", errors="coerce")
        df = df.dropna()

        df_res = df.resample('600s').mean()
        df_res.index = df_res.index.total_seconds().astype(int)

        simulated[col] = (df_res["T"] / 100).reindex(simulated.index)

    # ---------- Moisture ----------
    moisture = {
        "theta_8n": "obspt_RE_matrix-1.out",
        "theta_23n": "obspt_RE_matrix-3.out"
    }

    for col, filename in moisture.items():
        fpath = os.path.join(run_dir, "out", filename)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {filename}")

        df = pd.read_csv(
            fpath,
            comment='#',
            sep=r'\s+',
            header=None,
            skiprows=10,
            engine='python',
            names=[
                't', 'h', 'theta_l', 'theta_v',
                'l_flux', 'cum_flux', 'v_flux',
                'cum_l_flux', 'tot_flux',
                'total_flux', 'cum_flux2'
            ]
        )

        df = df.set_index('t')
        df.index = df.index.astype(float).round().astype(int)
        df.index = pd.to_timedelta(df.index, unit="s", errors="coerce")
        df = df.dropna()

        df_res = df.resample('600s').mean()
        df_res.index = df_res.index.total_seconds().astype(int)

        simulated[col] = df_res["theta_l"].reindex(simulated.index)

    diff = simulated[column_names[1:]] - measured[column_names[1:]]

    error = np.sqrt(np.nansum(diff.values ** 2))

    if not np.isfinite(error):
        raise ValueError("Non-finite error")

    return error


# ==============================
# DRUTES RUNNER
# ==============================

def runDrutes(par):
    # unpack parameters
    b1_org, b2_org, b3_org = par[0:3]
    b1_min, b2_min, b3_min = par[3:6]

    alpha_org, n_org, K_org = par[6:9]
    alpha_min, n_min, K_min = par[9:12]

    m_org = 1 - 1 / n_org
    m_min = 1 - 1 / n_min

    # call counter
    with counter_lock:
        global_counter.value += 1
        call_id = global_counter.value

    # node-local scratch
    scratch = os.environ.get("TMPDIR", "/tmp")
    run_id = uuid4().hex
    run_dir = os.path.join(scratch, f"drutes_run_{run_id}")

    cmd = [
        "bash", "run_drutes_parallel.sh", run_dir,
        str(b1_org), str(b2_org), str(b3_org),
        str(b1_min), str(b2_min), str(b3_min),
        str(alpha_org), str(n_org), str(m_org), str(K_org),
        str(alpha_min), str(n_min), str(m_min), str(K_min)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # early abort
        if not os.path.exists(os.path.join(run_dir, "out")):
            raise RuntimeError("No output directory")

        error = getError(run_dir)

    except Exception as e:
        print(f"[FAILED] run {run_id}: {e}")
        error = FAIL_PENALTY

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    log_run(call_id, error, par)
    return error


# ==============================
# MAIN
# ==============================

if __name__ == '__main__':

    bounds = [
        (0.02, 1.0), (0.02, 6.0), (0.02, 4.0),
        (0.02, 1.0), (0.02, 6.0), (0.02, 4.0),
        (0.15, 2000), (1.1, 5.0), (0.000864, 864),
        (0.15, 2000), (1.1, 5.0), (0.000864, 864)
    ]

    result = differential_evolution(
        runDrutes,
        bounds,
        strategy="rand1bin",
        popsize=20,
        mutation=(0.6, 1.0),
        recombination=0.7,
        tol=1e-4,
        maxiter=3000,
        workers=64,
        updating="deferred",
        polish=False
    )

    print("\nOPTIMIZED PARAMETERS:")
    print(result.x)
    print("OBJECTIVE FUNCTION:", result.fun)

