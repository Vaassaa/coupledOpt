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

def calculate_event_metrics(df, tree_name):
    # 1. Total Root Uptake (Integrate rate over time)
    # Assuming the rate is per second, we integrate against time (in seconds)
    time_delta_sec = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    uptake_total = integrate.trapezoid(df[f'{tree_name}_root'], x=time_delta_sec)
    
    # 2. Net Soil Retention (Storage at end - Storage at start)
    # Using your 'change' columns
    retention_total = df[f'{tree_name}_change'].iloc[-1] - df[f'{tree_name}_change'].iloc[0]
    
    return uptake_total, retention_total

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

def get_precipitation(file_path, start_filter, end_filter, reference_date):
    # Load data: rate is in m/s
    df = pd.read_csv(file_path, comment='#', sep='\\s+', names=['t', 'precip_rate'])
    
    # CONVERSION: Convert rate (m/s) to depth (m) for the 600s interval
    df['precip'] = df['precip_rate'] * 600
    
    df['datetime'] = df['t'].map(lambda x: reference_date + timedelta(seconds=int(x)))
    
    # Filter
    mask = (df['datetime'] >= start_filter) & (df['datetime'] <= end_filter)
    return df.loc[mask, ['datetime', 'precip']]
if __name__ == '__main__':
    # Which run directory to read, with a cache keyed to it so re-pointing never
    # reuses a stale cache (distinct name from getWaterVolume.py's cache).
    RUN_SUB = "drutes_run_best"
    cache_file = f"arch/processed_swc_data_{RUN_SUB}.csv"


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
        for df in [sim_beech, sim_spruce, meas_beech, meas_spruce]:
            df['datetime'] = df.index.map(lambda x: reference_date + timedelta(seconds=x))

        # 3. Filter
        start_filter = datetime(2024, 9, 13, 0, 0, 0)
        end_filter = datetime(2024, 9, 15, 0, 0, 0)
        sim_beech_range = sim_beech[(sim_beech["datetime"] >= start_filter) & (sim_beech["datetime"] <= end_filter)]
        sim_spruce_range = sim_spruce[(sim_spruce["datetime"] >= start_filter) & (sim_spruce["datetime"] <= end_filter)]
        meas_beech_range = meas_beech[(meas_beech["datetime"] >= start_filter) & (meas_beech["datetime"] <= end_filter)]
        meas_spruce_range = meas_spruce[(meas_spruce["datetime"] >= start_filter) & (meas_spruce["datetime"] <= end_filter)]
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

        # Read precipitation from the same config the model actually used so the
        # table throughfall is always consistent with the cached simulation data.
        # rain_free has no model run directory so it stays in the setup location.
        meteo_dir = "setup/dataIN/meteo/Campaing_2024-09-08_2024-09-30/"
        rain_beech = get_precipitation(f"arch/beech/{RUN_SUB}/drutes.conf/evaporation/rain.in", start_filter, end_filter, reference_date)
        rain_spruce = get_precipitation(f"arch/spruce/{RUN_SUB}/drutes.conf/evaporation/rain.in", start_filter, end_filter, reference_date)
        rain_free = get_precipitation(os.path.join(meteo_dir, 'rain_free.in'), start_filter, end_filter, reference_date)


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

        # Create a baseline (the storage at the start of your filter)
        baseline_beech = total_storage_beech.iloc[0]
        baseline_spruce = total_storage_spruce.iloc[0]

        # 5. Build Comparison DF
        comp_df = pd.DataFrame({
            'time': sim_beech_range["datetime"],
            'Beech_theta_meas': meas_beech_range["theta_8n"],
            'Spruce_theta_meas': meas_spruce_range["theta_8n"],
            'Beech_theta': sim_beech_range["theta_l_8n"],
            'Spruce_theta': sim_spruce_range["theta_l_8n"],
            'Beech': total_storage_beech,
            'Spruce': total_storage_spruce,
            'Spruce_root': root_uptake_spruce,
            'Beech_root': root_uptake_beech,
            'Rain_Beech': rain_beech['precip'].values,
            'Rain_Spruce': rain_spruce['precip'].values,
            'Rain_Free':rain_free['precip'].values,
            'Beech_change': (total_storage_beech - baseline_beech),
            'Spruce_change': (total_storage_spruce - baseline_spruce)

        })

        # 6. Save to CSV for next time
        comp_df.to_csv(cache_file, index=False)
        print(f"--- Data saved to {cache_file} ---")

    
    print(comp_df["Beech_root"])

    # Calculate total precipitation for each category (in mm)
    total_rain_beech = comp_df['Rain_Beech'].sum()
    total_rain_spruce = comp_df['Rain_Spruce'].sum()
    total_rain_free = comp_df['Rain_Free'].sum()

    print(f"Total Free Precipitation: {total_rain_free:.4f} m (i.e., {total_rain_free*1000:.1f} mm)")

    print("--- Total Precipitation (m) ---")
    print(f"Beech Throughfall: {total_rain_beech:.8f} m")
    print(f"Spruce Throughfall: {total_rain_spruce:.8f} m")
    print(f"Free Precipitation: {total_rain_free:.8f} m")
    print("--- Total Retention (m) ---")
    print(f"Beech soil retention: {np.max(comp_df["Beech_change"])/100:.8f} m")
    print(f"Beech root uptake: {np.sum(comp_df["Beech_root"]):.8f} m")
    print(f"Spruce soil retention: {np.max(comp_df["Spruce_change"])/100:.8f}")
    print(f"Spruce root uptake: {np.sum(comp_df["Spruce_root"]):.8f} m")


    # --- Event totals (consistent with the "Total Retention" section above) ---
    # root_uptake_* are per-timestep uptake depths [m] -> SUM over the event.
    # (The previous integrate.simpson over the seconds index multiplied the
    #  per-step values by the time axis, inflating them ~100-600x.)
    total_root_uptake_beech = np.sum(comp_df["Beech_root"])
    total_root_uptake_spruce = np.sum(comp_df["Spruce_root"])

    # *_change is cumulative storage change in cm -> max over the event, then /100
    # for metres (the previous .iloc[-1] without /100 reported cm as metres).
    total_retention_beech = np.max(comp_df['Beech_change']) / 100
    total_retention_spruce = np.max(comp_df['Spruce_change']) / 100

    print("\n" + "="*30)
    print("EVENT WATER BALANCE (m)")
    print("="*30)
    print(f"Beech Root Uptake:    {total_root_uptake_beech:.8f} m")
    print(f"Spruce Root Uptake:   {total_root_uptake_spruce:.8f} m")
    print("-"*30)
    print(f"Beech Soil Retention: {total_retention_beech:.8f} m")
    print(f"Spruce Soil Retention:{total_retention_spruce:.8f} m")
    print("="*30)
    
    """
    FIGURE
    """
    # Convert loaded time back to datetime
    pd.to_datetime(comp_df["time"])

    # 1. Define consistent color/style mapping
    colors = {'Free': 'slategray', 'Beech': 'forestgreen', 'Spruce': 'darkorange'}
    styles = {'Simulated': '-', 'Measured': '--'}

    # 2. Use gridspec_kw to make the rain subplot (index 0) shorter
    fig, (ax_rain, ax_moisture, ax_retention) = plt.subplots(
        nrows=3, ncols=1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={'height_ratios': [1, 2, 2]}
    )

    # --- Rain Subplot ---
    width = 0.01
    ax_rain.bar(comp_df['time'], comp_df['Rain_Free'], width=width, label='Open landscape', color=colors['Free'], alpha=0.3)
    ax_rain.bar(comp_df['time'], comp_df['Rain_Beech'], width=width, label='Beech throughfall', color=colors['Beech'], alpha=0.5)
    ax_rain.bar(comp_df['time'], comp_df['Rain_Spruce'], width=width, label='Spruce throughfall', color=colors['Spruce'], alpha=0.5)
    ax_rain.set_ylabel("Precipitation [m]")
    ax_rain.grid(True, linestyle='--', alpha=0.6)
    ax_rain.legend(loc='lower right', fontsize='small')

    # --- Moisture Subplot ---
    linewidth = 0.5
    ax_moisture.plot(comp_df["time"], comp_df["Beech_theta"], label='Beech (Simulated)', color=colors['Beech'],linestyle=styles['Simulated'])
    ax_moisture.plot(comp_df["time"], comp_df["Spruce_theta"], label='Spruce (Simulated)', color=colors['Spruce'], linestyle=styles['Simulated'])
    ax_moisture.plot(comp_df["time"], comp_df["Beech_theta_meas"], label='Beech (Measured)', color=colors['Beech'],
                     linestyle=styles['Measured'], alpha=linewidth)
    ax_moisture.plot(comp_df["time"], comp_df["Spruce_theta_meas"], label='Spruce (Measured)', color=colors['Spruce'],
                     linestyle=styles['Measured'], alpha=linewidth)
    ax_moisture.set_ylabel("Soil Moisture [-]")
    ax_moisture.grid(True, linestyle='--', alpha=0.6)
    ax_moisture.legend(loc='lower right', fontsize='small')

    # --- Retention Subplot ---
    ax_retention.plot(comp_df["time"], comp_df['Beech_change']/100, label='Beech soil', color=colors['Beech'], marker='o', markersize=2)
    ax_retention.plot(comp_df["time"], comp_df['Spruce_change']/100, label='Spruce soil', color=colors['Spruce'], marker='o', markersize=2)
    ax_retention.set_ylabel("Accumulated Water [m]")
    ax_retention.legend(loc='lower right', fontsize='small')
    ax_retention.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("figs/water_content_comparison.png", dpi=300)
    plt.show()

    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # # Axis 1: Total Water Storage
    # ax1.plot(comp_df["time"], comp_df['Spruce']/100, label='Total Storage', color='darkorange')
    # ax1.set_ylabel("Total Storage (m)", color='darkorange')

    # # Axis 2: Root Uptake
    # ax2 = ax1.twinx()
    # ax2.plot(comp_df["time"], comp_df['Spruce_root'], label='Cumulative Uptake', color='red', linestyle='-', marker="o")
    # ax2.set_ylabel("Cumulative Uptake (m)", color='red')

    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    # plt.title("Spruce: Storage vs. Root Uptake")
    # plt.show()
