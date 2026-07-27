#!/usr/bin/env python3
"""
Morris (elementary effects) sensitivity screening for the coupled Saito-Sakai
beech calibration.

Why this exists
---------------
`log_analysis.py` computes a Spearman correlation over the points the DE
optimiser happened to sample. That is a biased, adaptive sample: as DE
converges it squeezes the influential parameters into narrow ranges, so the
correlation is confounded with the optimiser's state and is only a rough
screening heuristic.

This script instead performs a *designed* Morris one-at-a-time (OAT) sampling
of the full parameter space and computes proper, sampling-independent
sensitivity measures:

    mu_star  — overall influence (mean of |elementary effects|); the ranking metric
    sigma    — non-linearity / interaction (std of elementary effects)
    mu       — signed mean effect (direction: does raising the param raise error?)

It reuses the EXACT DRUtES evaluation pipeline used by the calibration:
`run_drutes.sh` + `getError()` from run_calibration_beech.py. The parameter
mapping below mirrors `runDrutes(strategy="all", ...)` in that file — keep the
two in sync if you change the parameterisation. It deliberately does NOT log to
de_log_beech.csv, so your DE log stays clean.

Sampling happens in the same log10-transformed space the DE searches, so a unit
of a `log10_*` parameter is one decade, and mu_star for those params is "error
change per decade".

Usage
-----
    python morris_screening.py [--trajectories N] [--levels L] [--workers W]
                               [--target {error,error_heat,error_moist}] [--seed S]

Total DRUtES runs = trajectories * (num_params + 1) = N * 15.
    N=20  -> 300 runs  (screening; mu_star ranking usually stable here)
    N=40  -> 600 runs  (firmer mu_star_conf confidence intervals)

Outputs
-------
    morris_samples_beech.csv   raw design + all three error components (re-analyse offline)
    morris_results_beech.csv   mu, mu_star, mu_star_conf, sigma per parameter (ranked)
    morris_mustar_sigma_beech.png   the standard mu_star-vs-sigma influence plot
    morris_ranking_beech.png        ranked mu_star bar chart with confidence bars

Author: Vaclav Steinbach
"""
import os
import sys
import argparse
import shutil
import subprocess
import multiprocessing
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# --------------------------------------------------------------------------- #
# Import the exact evaluation helpers from the calibration script.            #
# run_calibration_beech.py has a module-level `stage = sys.argv[1]`, so we    #
# stash the real argv, hand it a dummy stage for the import, then restore.    #
# --------------------------------------------------------------------------- #
_REAL_ARGV = sys.argv[:]
sys.argv = [_REAL_ARGV[0], "broad"]
import run_calibration_beech as cal          # provides getError, calcHydraulicHead, DRUTES_TEMPLATE
sys.argv = _REAL_ARGV

# --------------------------------------------------------------------------- #
# Parameter space — MUST mirror the "broad" bounds and the "all" unpacking in #
# run_calibration_beech.py. log10_* entries are searched (and sampled) in     #
# log10 space, exactly like the DE.                                           #
# --------------------------------------------------------------------------- #
NAMES = [
    "b1_org", "b2_org", "b3_org",
    "b1_min", "b2_min", "b3_min",
    "albedo",
    "log10_alpha_org", "n_org", "log10_K_org",
    "log10_alpha_min", "n_min", "log10_K_min",
    "log10_S_max",
]

# NOTE: these are FEASIBLE SCREENING bounds, deliberately narrower than the DE
# "broad" bounds in run_calibration_beech.py. Morris samples the box uniformly,
# so extreme corners the DE never visits caused ~60% of runs to fail (initial
# head h_min<-500 for n_min near 1.05; DRUtES non-convergence for K_org<~1e-5).
# Two lower bounds were raised to physically-plausible, numerically-stable values
# so the elementary effects are computed over a mostly-feasible domain. If you
# widen these back out, expect the failure rate (and penalty pollution) to rise.
BOUNDS = [
    [0.02, 2.0],                                          # b1_org
    [0.02, 8.0],                                          # b2_org
    [0.02, 6.0],                                          # b3_org
    [0.02, 2.0],                                          # b1_min
    [0.02, 8.0],                                          # b2_min
    [0.02, 6.0],                                          # b3_min
    [0.05, 0.3],                                          # albedo
    [float(np.log10(1)),      float(np.log10(10))],       # log10_alpha_org
    [2.25, 5.0],                                          # n_org
    [float(np.log10(1.0e-5)), float(np.log10(10.0e-4))],  # log10_K_org  (raised from 1e-8: <1e-5 fails to converge)
    [float(np.log10(1)),      float(np.log10(10))],       # log10_alpha_min
    [1.30, 2.0],                                          # n_min        (raised from 1.05: n_min<1.3 -> h_min<-500 IC rejection)
    [float(np.log10(1.0e-8)), float(np.log10(10.0e-4))],  # log10_K_min
    [float(np.log10(1e-9)),   float(np.log10(10e-7))],    # log10_S_max
]

PROBLEM = {"num_vars": len(NAMES), "names": NAMES, "bounds": BOUNDS}


def evaluate(par):
    """
    Run one DRUtES simulation for a 14-vector `par` (in DE/log10 space) and
    return (error, error_heat, error_moist, status). Mirrors runDrutes("all", ...).

    status is one of:
        "ok"        — simulation succeeded, errors are real
        "ic_reject" — infeasible initial head (h < -500 m); errors are NaN
        "crash"     — DRUtES crashed or timed out; errors are NaN
    so the caller can report the failure breakdown and treat causes distinctly.
    """
    # ---- parameter mapping: mirrors runDrutes strategy=="all" ---------------
    b1_org, b2_org, b3_org = par[0], par[1], par[2]
    b1_min, b2_min, b3_min = par[3], par[4], par[5]
    albedo = par[6]
    alpha_org = 10 ** par[7]
    n_org = par[8]
    K_org = 10 ** par[9]
    alpha_min = 10 ** par[10]
    n_min = par[11]
    K_min = 10 ** par[12]
    S_max = 10 ** par[13]

    # ---- initial-condition sanity check (same as calibration) ---------------
    monitoring = pd.read_csv(
        cal.DRUTES_TEMPLATE + "/drutes.conf/inverse_modeling/monitoring.dat",
        comment="#", sep=r"\s+", header=None,
    )
    theta_org = monitoring.iloc[0, 4]
    theta_min = monitoring.iloc[0, 5]
    h_org = cal.calcHydraulicHead(theta_org, [alpha_org, n_org, 1 - 1 / n_org])
    h_min = cal.calcHydraulicHead(theta_min, [alpha_min, n_min, 1 - 1 / n_min])
    if h_org < -500 or h_min < -500:
        return (np.nan, np.nan, np.nan, "ic_reject")

    run_dir = f"morris_run_{uuid4().hex}/"
    cmd = [
        "bash", "run_drutes.sh", cal.DRUTES_TEMPLATE, run_dir,
        str(b1_org), str(b2_org), str(b3_org),
        str(b1_min), str(b2_min), str(b3_min),
        str(albedo),
        str(alpha_org), str(n_org), str(1 - 1 / n_org), str(K_org),
        str(alpha_min), str(n_min), str(1 - 1 / n_min), str(K_min),
        str(S_max),
        str(h_org), str(h_min),
    ]
    try:
        subprocess.run(cmd, timeout=900, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        e, eh, em = cal.getError(run_dir)     # (error, error_heat, error_moist)
        return (e, eh, em, "ok")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return (np.nan, np.nan, np.nan, "crash")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def make_plots(names, res_df, target):
    """Standard Morris covariance plot + ranked mu_star bar chart."""
    # mu_star vs sigma
    fig, ax = plt.subplots()
    ax.scatter(res_df["mu_star"], res_df["sigma"], color="black", zorder=3)
    for _, r in res_df.iterrows():
        ax.annotate(r["parameter"], (r["mu_star"], r["sigma"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    lim = max(res_df["mu_star"].max(), res_df["sigma"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="grey")       # sigma = mu_star
    ax.set_xlabel(r"$\mu^*$ (overall influence on " + target + ")")
    ax.set_ylabel(r"$\sigma$ (non-linearity / interactions)")
    ax.set_title(f"Morris screening — beech — target: {target}")
    fig.tight_layout()
    fig.savefig("morris_mustar_sigma_beech.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ranked mu_star with confidence bars
    ranked = res_df.sort_values("mu_star", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(ranked["parameter"], ranked["mu_star"],
            xerr=ranked["mu_star_conf"], color="0.4", ecolor="black", capsize=3)
    ax.set_xlabel(r"$\mu^*$ (target: " + target + ")")
    ax.set_title("Morris parameter ranking — beech")
    fig.tight_layout()
    fig.savefig("morris_ranking_beech.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Morris sensitivity screening (beech).")
    p.add_argument("--trajectories", type=int, default=20,
                   help="Morris trajectories r; total runs = r*(k+1) = r*15 (default 20).")
    p.add_argument("--levels", type=int, default=4,
                   help="Morris grid levels p (default 4; must match sample & analyze).")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                   help="Parallel DRUtES processes (default = available CPUs).")
    p.add_argument("--target", default="error",
                   choices=["error", "error_heat", "error_moist"],
                   help="Which error component to analyse (default total error).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = p.parse_args()

    # 1. Design ------------------------------------------------------------- #
    X = morris_sample.sample(PROBLEM, N=args.trajectories,
                             num_levels=args.levels, seed=args.seed)
    print(f"Morris design: {len(X)} DRUtES runs "
          f"({args.trajectories} trajectories x {PROBLEM['num_vars']+1}), "
          f"{args.workers} workers.")

    # 2. Evaluate (parallel, order preserved by ex.map) --------------------- #
    # Force fork so worker processes inherit the already-imported `cal` module.
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        results = list(ex.map(evaluate, X))
    Y_all = np.asarray([r[:3] for r in results], dtype=float)   # (n, 3)
    status = [r[3] for r in results]

    # 2b. Report failure breakdown by cause -------------------------------- #
    n_ok = status.count("ok")
    n_ic = status.count("ic_reject")
    n_crash = status.count("crash")
    print(f"Runs: {n_ok} ok | {n_ic} IC-rejected (h<-500) | {n_crash} crash/timeout "
          f"({(n_ic + n_crash) / len(status):.0%} failed)")
    if (n_ic + n_crash) / len(status) > 0.2:
        print("  NOTE: >20% failed -> penalty substitution biases the ranking. "
              "Tighten the sampling bounds (see BOUNDS comment) before trusting results.")

    # 3. Persist raw design + outputs (full transparency / offline re-analysis)
    samples = pd.DataFrame(X, columns=NAMES)
    samples["error"] = Y_all[:, 0]
    samples["error_heat"] = Y_all[:, 1]
    samples["error_moist"] = Y_all[:, 2]
    samples["status"] = status
    samples.to_csv("morris_samples_beech.csv", index=False)

    # 4. Failure handling for the chosen target ----------------------------- #
    tgt_idx = {"error": 0, "error_heat": 1, "error_moist": 2}[args.target]
    Y = Y_all[:, tgt_idx].copy()
    nfail = int(np.isnan(Y).sum())
    if nfail:
        penalty = np.nanmax(Y) * 1.2
        print(f"WARNING: {nfail}/{len(Y)} runs failed (rejected IC / crash / timeout). "
              f"Substituting penalty {penalty:.4g}. Elementary effects touching these "
              f"points are approximate — inspect morris_samples_beech.csv and consider "
              f"re-running if many trajectories are affected.")
        Y = np.where(np.isnan(Y), penalty, Y)

    # 5. Analyse ------------------------------------------------------------ #
    Si = morris_analyze.analyze(PROBLEM, X, Y, num_levels=args.levels,
                                print_to_console=False, seed=args.seed)
    res = pd.DataFrame({
        "parameter": Si["names"],
        "mu": Si["mu"],
        "mu_star": Si["mu_star"],
        "mu_star_conf": Si["mu_star_conf"],
        "sigma": Si["sigma"],
    }).sort_values("mu_star", ascending=False).reset_index(drop=True)
    res.to_csv("morris_results_beech.csv", index=False)

    print("\n" + "=" * 62)
    print(f"MORRIS SENSITIVITY — BEECH — target: {args.target}")
    print("=" * 62)
    print(res.to_string(index=False,
                        formatters={c: "{:.4g}".format
                                    for c in ["mu", "mu_star", "mu_star_conf", "sigma"]}))
    print("-" * 62)
    print("Ranked by mu_star (overall influence). A parameter is 'influential'")
    print("when mu_star clearly exceeds mu_star_conf (its confidence half-width).")
    print("High sigma relative to mu_star => strong non-linearity / interactions.")
    print("=" * 62)

    make_plots(NAMES, res, args.target)
    print("Saved: morris_samples_beech.csv, morris_results_beech.csv,")
    print("       morris_mustar_sigma_beech.png, morris_ranking_beech.png")


if __name__ == "__main__":
    main()
