"""
Collection of small helper functions for
calibration of coupled Saito-Sakai
soil moisture and temperature transport model
Author: Vaclav Steinbach
Date: 06.03.2026
Dissertation
"""
from datetime import datetime
import numpy as np

def log_run(call_id, error, error_heat, error_moist, par, logfile="de_log.csv"):
    """
    Appends a single optimisation step to a CSV log file.
    -----------------------------------------------------
    Records the call ID, timestamp, combined and component errors (heat/moisture),
    and the full parameter vector. Creates or extends the log file on each call.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{call_id},"
        f"{timestamp},"
        f"{error:.8g},"
        f"{error_heat:.8g},"
        f"{error_moist:.8g},"
        + ",".join(f"{p:.8g}" for p in par)
        + "\n"
    )
    with open(logfile, "a") as f:
        f.write(line)

def calcHydraulicHead(theta, ret_par):
    """
    Calculates hydraulic head "h" from retenction curve parameters given
    intial soil moisture "theta"
    --------------------------------------------------------------------
    Used in a if clause for early stopping of simulations with unrealistic
    intial condition.
    """
    alpha = ret_par[0] 
    n     = ret_par[1] 
    m     = ret_par[2]
    h = - 1/alpha * (1 - (theta**(1/m)) / (theta**(1/m)))**(1/n)
    return h


def shrink_bounds(x, bounds, shrink=0.25):
    """
    Narrows parameter bounds symmetrically around a given solution vector.
    ----------------------------------------------------------------------
    For each parameter, the search interval is contracted to a fraction
    (`shrink`) of its original span centred on x, without exceeding the
    original bounds. Useful for local refinement after a global search.
    """
    new_bounds = []
    for xi, (lo, hi) in zip(x, bounds):
        span = hi - lo
        new_lo = max(lo, xi - shrink * span)
        new_hi = min(hi, xi + shrink * span)
        new_bounds.append((new_lo, new_hi))
    return new_bounds


def jitter_init(x, bounds, rel=0.05, size=16):
    """
    Generates an initial population by randomly perturbing a reference vector.
    --------------------------------------------------------------------------
    Each of the `size` candidates is produced by adding uniform noise scaled
    to `rel` * parameter span. All values are clipped to the supplied bounds.
    Intended to seed a differential evolution run near a known good solution.
    """
    pop = []
    for _ in range(size):
        xi = []
        for v, (lo, hi) in zip(x, bounds):
            span = hi - lo
            dv = np.random.uniform(-rel, rel) * span
            xi.append(np.clip(v + dv, lo, hi))
        pop.append(xi)
    return np.array(pop)
