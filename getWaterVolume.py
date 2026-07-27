import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from calibration_tools import calcHydraulicHead
import sys
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy import integrate 
import os

def getData(run_dir):
    column_names = ["time", "T_8n", "T_15n", "T_23n", "theta_8n", "theta_23n"]
    
    start_warming = datetime(2024, 9, 8, 0, 0, 0)
    end_warming = datetime(2024, 9, 12, 0, 0, 0)
    prewarming_sec = (end_warming - start_warming).total_seconds()

    # Load measured
    meas_path = os.path.join(run_dir, 'drutes.conf/inverse_modeling/monitoring.dat')
    measured = pd.read_csv(meas_path, comment='#', sep='\\s+', header=None, names=column_names, index_col='time')
    
    all_sim_cols = []
    observation_points = 150 

    # --- Process Files ---
    # Combine the two loops into one to avoid double-reading logic
    for i in range(1, observation_points):
        print(f"Processing obspt_RE_matrix-{i}.out")
        # Moisture
        m_path = os.path.join(run_dir, f'out/obspt_RE_matrix-{i}.out')
        if os.path.exists(m_path):
            df = pd.read_csv(m_path, comment='#', sep='\\s+', header=None, skiprows=10, engine='c',
                             names=['t','h','theta_l','theta_v','sat_degree','l_flux','cum_l_flux','v_flux','cum_v_flux','total_flux','cum_total_flux'])
            df.index = pd.to_timedelta(df['t'].astype(float).round().astype(int), unit="s")
            df = df[~df.index.duplicated(keep='last')].dropna()
            df_res = df.resample('600s').mean()
            df_res.index = df_res.index.total_seconds().astype(int)

            # Extract Flux
            s_flux = df_res["l_flux"].reindex(measured.index)
            s_flux.name = f"l_flux_{i}n"
            all_sim_cols.append(s_flux)
            
            # Extract Head (h)
            s_head = df_res["h"].reindex(measured.index)
            s_head.name = f"h_{i}n"
            all_sim_cols.append(s_head)

            # Extract Theta 
            s_theta = df_res["theta_l"].reindex(measured.index)
            s_theta.name = f"theta_l_{i}n"
            all_sim_cols.append(s_theta)

    simulated = pd.concat(all_sim_cols, axis=1)

    # Cutoff based on index (seconds)
    measured_cutoff = measured[measured.index > prewarming_sec]
    simulated_cutoff = simulated[simulated.index > prewarming_sec]

    return measured_cutoff, simulated_cutoff

def get_total_retention(df, columns):
    integrals = integrate.cumulative_trapezoid(df[columns], df.index, initial=0)
    return pd.Series(np.abs(integrals[-1]), index=columns)

def get_depth_idx(series):
    return [int(name.split('_')[2].replace('n', '')) for name in series.index]


# def get_feddes_alpha(h, h1, h2, h3, h4):
    # alpha = np.zeros_like(h)
    # # Range 1: h1 to h2 (Anaerobic to optimal)
    # mask1 = (h > h2) & (h < h1)
    # alpha[mask1] = (h1 - h[mask1]) / (h1 - h2)
    
    # # Range 2: h2 to h3 (Optimal)
    # mask2 = (h >= h3) & (h <= h2)
    # alpha[mask2] = 1.0
    
    # # Range 3: h3 to h4 (Wilting point)
    # mask3 = (h > h4) & (h < h3)
    # alpha[mask3] = (h[mask3] - h4) / (h3 - h4)
    
    # return alpha

def get_feddes_alpha(h, h1, h2, h3, h4):
    alpha = np.zeros_like(h, dtype=float)
    
    # Range 1: h1 to h2 (Anaerobic)
    if h1 > h2:
        mask1 = (h > h2) & (h < h1)
        alpha[mask1] = (h1 - h[mask1]) / (h1 - h2)
    else:
        # If h1 == h2, it's a step change
        alpha[h > h1] = 0.0
    
    # Range 2: h2 to h3 (Optimal)
    mask2 = (h >= h3) & (h <= h2)
    alpha[mask2] = 1.0
    
    # Range 3: h3 to h4 (Wilting)
    if h3 > h4:
        mask3 = (h > h4) & (h < h3)
        alpha[mask3] = (h[mask3] - h4) / (h3 - h4)
    else:
        # If h3 == h4, it's a step change
        alpha[h < h3] = 0.0
    
    return alpha

def calculate_root_uptake(df, params):
    """
    params: dict with keys h1, h2, h3, h4, Smax
    """
    h_cols = [col for col in df.columns if col.startswith('h_')]
    uptake_matrix = pd.DataFrame(index=df.index)
    
    for col in h_cols:
        # Extract depth from column name (e.g., 'h_8n' -> 8)
        depth = int(col.split('_')[1].replace('n', ''))
        
        # Apply parameters based on depth
        p = params['upper'] if depth <= 8 else params['lower']
        
        # Calculate alpha and then S(h)
        alpha = get_feddes_alpha(df[col].values, p['h1'], p['h2'], p['h3'], p['h4'])
        print(f"Col {col} | Mean Alpha: {alpha.mean():.4f} | Range: [{alpha.min():.2f}, {alpha.max():.2f}]")
        uptake_matrix[col] = alpha * p['Smax']
        
    # Integrate over depth (1cm slices)
    return uptake_matrix.sum(axis=1)

def get_soil_retention_profile(df, flux_cols):
    """
    Calculates the net water stored in each 1cm soil layer.
    Retention_layer_i = Integral(Flux_top) - Integral(Flux_bottom)
    """
    # 1. Integrate all flux columns over time to get Total Volume (m) passing through each depth
    # df.index is time in seconds
    integrated_flux = integrate.simpson(df[flux_cols], x=df.index, axis=0)
    
    # 2. Calculate the difference between adjacent layers
    # Retention in layer i = Integrated Flux at depth i - Integrated Flux at depth i+1
    # This represents the water that 'stayed' in that 1cm slice.
    soil_retention = np.diff(integrated_flux) 
    
    # Append a 0 or the last value to keep the length at 150
    soil_retention = np.append(soil_retention, 0)
    
    return np.abs(soil_retention) # Absolute value depends on your coordinate system (downward vs upward)

def calculate_storage_over_time(df):
    """
    df: Dataframe where columns are 'theta_l_1n', 'theta_l_2n', ...
    Assumes each column corresponds to 1cm depth (if your spacing is different, 
    multiply the integral result by your grid spacing).
    """
    # 1. Extract only the theta columns
    theta_cols = [col for col in df.columns if col.startswith('theta_l')]
    
    # 2. Get the depth values (if they are 1cm increments, this is just a constant)
    # If your depths are -1, -2, ..., -150, use those as the x-axis
    depths = [int(name.split('_')[2].replace('n', '')) for name in theta_cols]
    
    # 3. Integrate over depth for every row (time step)
    # axis=1 integrates across the columns (the depths)
    total_water_per_time = integrate.trapezoid(df[theta_cols].values, x=depths, axis=1)
    
    return pd.Series(total_water_per_time, index=df.index)

if __name__ == '__main__':
    # Which run directory to read (drutes_run or drutes_run_new) and a cache keyed
    # to it so re-pointing never reuses a stale cache.
    RUN_SUB = "drutes_run_new"
    cache_file = f"arch/processed_retention_data_{RUN_SUB}.csv"


    if os.path.exists(cache_file):
        print(f"--- Loading cached data from {cache_file} ---")
        comp_df = pd.read_csv(cache_file)
        # --- CRITICAL FIX: Re-convert string column to datetime ---
        comp_df['time'] = pd.to_datetime(comp_df['time'])
    
    else:
        print("--- No cache found. Starting full processing ---")
        
        # 1. Load Data
        print(f"--- Processing beech ---")
        meas_beech, sim_beech = getData(f"arch/beech/{RUN_SUB}/")
        print(f"--- Processing spruce ---")
        meas_spruce, sim_spruce = getData(f"arch/spruce/{RUN_SUB}/")

        # 2. Add Datetime objects for filtering
        reference_date = datetime(2024, 9, 8, 0, 0, 0)
        for df in [sim_beech, sim_spruce]:
            df['datetime'] = df.index.map(lambda x: reference_date + timedelta(seconds=x))

        # 3. Filter
        start_filter = datetime(2024, 9, 13, 3, 0, 0)
        end_filter = datetime(2024, 9, 15, 12, 0, 0)
        sim_beech_range = sim_beech[(sim_beech["datetime"] >= start_filter) & (sim_beech["datetime"] <= end_filter)]
        sim_spruce_range = sim_spruce[(sim_spruce["datetime"] >= start_filter) & (sim_spruce["datetime"] <= end_filter)]
        print(sim_beech_range)

        # 4. Process Retention
        flux_cols = [col for col in sim_beech_range.columns if col.startswith('theta_l')]
        print(flux_cols)
        total_storage_beech = calculate_storage_over_time(sim_beech_range)
        total_storage_spruce = calculate_storage_over_time(sim_spruce_range)


        spruce_params = {
            'upper': {'h1': -0.1, 'h2': -0.1, 'h3': -8, 'h4': -150, 'Smax': 5.3750156e-09},
            'lower': {'h1': -0.1, 'h2': -0.1, 'h3': -8, 'h4': -150, 'Smax': 5.3750156e-09}
        }
        beech_params = {
            'upper': {'h1': 0, 'h2': 0, 'h3': -9, 'h4': -150, 'Smax': 1.79521e-08},
            'lower': {'h1': 0, 'h2': 0, 'h3': -9, 'h4': -150, 'Smax': 1.79521e-08}
        }

        root_uptake_spruce = calculate_root_uptake(sim_spruce_range, spruce_params)
        root_uptake_beech = calculate_root_uptake(sim_beech_range, beech_params)

        # ret_beech = get_total_retention(sim_beech_range, flux_cols)
        # ret_spruce = get_total_retention(sim_spruce_range, flux_cols)

        # 1. Total Soil Retention (Net change in storage)
        # We take what entered at the surface and subtract what left the bottom
        # total_soil_beech = np.abs(ret_beech.iloc[0] - ret_beech.iloc[-1])

        # Calculate the profiles (one value per horizon)
        # tree_profile_beech = get_retention_profile(sim_beech_range, h_cols, beech_params, beech_smax)
        # tree_profile_spruce = get_retention_profile(sim_spruce_range, h_cols, spruce_params, spruce_smax)

        # # 5. Build Comparison DF
        # comp_df = pd.DataFrame({
            # 'depth_index': get_depth_idx(ret_beech),
            # 'Beech': total_storage_beech,
            # 'Spruce': total_storage_spruce
        # }).sort_values('depth_index')

        # 5. Build Comparison DF
        comp_df = pd.DataFrame({
            'time': sim_beech_range["datetime"],
            'Beech': total_storage_beech,
            'Spruce': total_storage_spruce,
            'Spruce_root': root_uptake_spruce,
            'Beech_root': root_uptake_beech
        })

        # 6. Save to CSV for next time
        comp_df.to_csv(cache_file, index=False)
        print(f"--- Data saved to {cache_file} ---")

    
    print(comp_df["Beech_root"])

    # 7. Plotting as shared x-axis subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    pd.to_datetime(comp_df["time"])
    # Plot Beech on the top subplot
    ax1.plot(comp_df["time"], comp_df['Beech']/100, 
             label='Beech', color='royalblue', marker='o', markersize=2)
    ax1.plot(comp_df["time"], comp_df['Beech_root'], 
             label='Beech root', color='royalblue', linestyle='--', markersize=2)
    ax1.set_ylabel("Beech Water (m)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Spruce on the bottom subplot
    ax2.plot(comp_df["time"], comp_df['Spruce']/100, 
             label='Spruce', color='darkorange', marker='o', markersize=2)
    ax2.plot(comp_df["time"], comp_df['Spruce_root']/100, 
             label='Spruce_root', color='darkorange', linestyle='--', markersize=2)
    ax2.set_ylabel("Spruce Water (m)")
    ax2.set_xlabel("Time")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("figs/water_content_comparison.png", dpi=300)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: Total Water Storage
    ax1.plot(comp_df["time"], comp_df['Spruce']/100, label='Total Storage', color='darkorange')
    ax1.set_ylabel("Total Storage (m)", color='darkorange')

    # Axis 2: Root Uptake
    ax2 = ax1.twinx()
    ax2.plot(comp_df["time"], comp_df['Spruce_root'], label='Cumulative Uptake', color='red', linestyle='-', marker="o")
    ax2.set_ylabel("Cumulative Uptake (m)", color='red')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Spruce: Storage vs. Root Uptake")
    plt.show()
