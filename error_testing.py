import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

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
        # drop duplicates
        df = df[~df.index.duplicated(keep='last')]
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
        # drop duplicates
        df = df[~df.index.duplicated(keep='last')]
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
    
    # cutoff prewarming time from simulated data
    simulated_cutoff = simulated.query("time > @prewarming_sec")

    # Compute residuals
    diff = simulated_cutoff[column_names[1:]] - measured_cutoff[column_names[1:]]
    diff_norm = diff.copy()
    # Omit NaNs if present
    diff = diff.dropna()

    # Split by physics
    heat_cols = ["T_8n", "T_15n", "T_23n"]
    moisture_cols = ["theta_8n", "theta_23n"]

    # Calculate a Weight Vector based on the Intensity of the signal
    T_min = measured[heat_cols].min().min()
    T_max = measured[heat_cols].max().max()

    T_mean = measured[heat_cols].mean().mean()
    T_range = (T_max - T_min) / 2  # half-range as normalizer

    # Weight = 2.0 at mean, 4.0 at both extremes
    weights_T = 2.0 + ((measured[heat_cols] - T_mean).abs() / T_range)

    # Physical normalization based on signal variability
    sigma_T     = measured[heat_cols].std().mean()      # °C
    sigma_theta = measured[moisture_cols].std().mean()  # [-]
    print(f"\nNormalization constants:  σ_T = {sigma_T:.4f}   σ_θ = {sigma_theta:.4f}\n")

    # Physical normalization scales
    # sigma_T     = 1.0    # °C
    # sigma_theta = 0.05   # [-]

    # Weighted peaks and depths approach
    for col in heat_cols:
        diff_norm[col] = (diff[col] / sigma_T) * weights_T[col]

    for col in moisture_cols:
        diff_norm[col] = diff[col] / sigma_theta

    # # Physical normalization approach
    # for col in heat_cols:
        # diff_norm[col] = diff[col] / sigma_T

    # for col in moisture_cols:
        # diff_norm[col] = diff[col] / sigma_theta

    # Compute separate errors
    error_heat = np.sqrt(np.mean(diff_norm[heat_cols].values**2))
    error_moist = np.sqrt(np.mean(diff_norm[moisture_cols].values**2))

    # Combine
    # error = error_heat + error_moist
    heat_weight = 1.0
    moist_weight = 1.0
    error = np.sqrt((heat_weight*error_heat)**2 + (moist_weight*error_moist)**2) # quad aggregation
    return error, error_heat, error_moist, diff, diff_norm, simulated_cutoff, measured_cutoff

if __name__ == '__main__':
    sim_dir = 'drutes_run/'

    # compute error
    error, error_heat, error_moist, diff, diff_norm, simulated, measured = getError(sim_dir)
    print(f"ERROR: {error:.4f}\nERROR HEAT: {error_heat:.4f}\nERROR MOIST: {error_moist:.4f}")
    print(f"\n{'Variable':<12} {'Min':>10} {'Max':>10} {'Range':>10}")
    print(f"{'─'*44}")
    th = diff_norm["theta_8n"]
    T  = diff_norm["T_8n"]
    print(f"{'theta_8n diff':<12} {th.min():>10.4f} {th.max():>10.4f} {abs(th.min()-th.max()):>10.4f}")
    print(f"{'T_8n diff':<12} {T.min():>10.4f} {T.max():>10.4f} {abs(T.min()-T.max()):>10.4f}\n")


    fig, ax = plt.subplots(2,1)
    ax[0].plot(simulated['T_8n'], color='green', label='sim')
    ax[0].plot(measured['T_8n'], color='blue', label='mes')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(diff['T_8n'], color='red', label='diff')
    ax[1].plot(diff_norm['T_8n'], color='gray', label='diff_norm')
    ax[1].legend()
    ax[1].grid()
    fig.savefig("figs/error_analysis_T_fixed.png")
    # plt.show()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(simulated['theta_8n'], color='green', label='sim')
    ax[0].plot(measured['theta_8n'], color='blue', label='mes')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(diff['theta_8n'], color='red', label='diff')
    ax[1].plot(diff_norm['theta_8n'], color='gray', label='diff_norm')
    ax[1].legend()
    ax[1].grid()
    fig.savefig("figs/error_analysis_theta_fixed.png")
    # plt.show()



