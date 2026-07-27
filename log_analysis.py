import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

analysis_type = sys.argv[1]


def format_param_latex(col):
    """Strip the unit suffix and render a parameter name as LaTeX subscripts.

    e.g. 'b1_org[W/(m.K)]' -> '$b_{1_{org}}$', 'alpha_min[1/m]' -> '$\\alpha_{min}$'.
    Matplotlib mathtext forbids adjacent subscripts (b_{1}_{org}), so chained
    subscripts are nested instead.
    """
    greek = {'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
             'theta': r'\theta', 'lambda': r'\lambda', 'sigma': r'\sigma'}

    # Drop the bracketed unit, e.g. '[W/(m.K)]'
    name = re.sub(r'\[.*?\]', '', col).strip()

    parts = name.split('_')
    head = parts[0]
    # Split a trailing number off the base, e.g. 'b1' -> 'b', '1'
    m = re.match(r'^([A-Za-z]+)(\d+)$', head)
    if m:
        base, subs = m.group(1), [m.group(2)] + parts[1:]
    else:
        base, subs = head, parts[1:]

    base = greek.get(base, base)

    if not subs:
        return f'${base}$'

    # Nest subscripts so multiple levels render: base_{s1_{s2_{...}}}
    sub_str = subs[-1]
    for s in reversed(subs[:-1]):
        sub_str = f'{s}_{{{sub_str}}}'
    return f'${base}_{{{sub_str}}}$'

# 1. Load the data, skipping the comment line
# df = pd.read_csv('logs/de_log_spruce.csv', comment='#')
df = pd.read_csv('de_log_beech.csv', comment='#')
df.columns = df.columns.str.strip()

match analysis_type:
    case "minimum":
        error_col = df["error"]
        min_error_idx = error_col.idxmin()
        min_error_val = error_col[min_error_idx]
        print(f"Best aggregated error found in log: {min_error_val:.6f} (row {min_error_idx+4})")
        error_col = df["error_heat"]
        min_error_idx = error_col.idxmin()
        min_error_val = error_col[min_error_idx]
        print(f"Best heat error found in log: {min_error_val:.6f} (row {min_error_idx+4})")
        error_col = df["error_moist"]
        min_error_idx = error_col.idxmin()
        min_error_val = error_col[min_error_idx]
        print(f"Best moist error found in log: {min_error_val:.6f} (row {min_error_idx+4})")

    case "sensitivity":
        # 2. Linearize the log10 parameters (alpha_org, alpha_min, K_min, K_org, S_max)
        # Note: Spearman correlation remains the same after this, but it's physically correct for your analysis
        # log_params = ['alpha_org[1/m]', 'alpha_min[1/m]', 'K_min[m/s]', 'K_org[m/s]', 'S_max[m/s]']
        # for col in log_params:
            # if col in df.columns:
                # df[col] = 10**df[col]

        # 3. Define target columns
        error_cols = ['error', 'error_heat', 'error_moist']
        exclude_cols = ['call_id', 'timestamp'] + error_cols
        param_cols = [c for c in df.columns if c not in exclude_cols]

        # 4. Compute Spearman correlation
        # Spearman (rank-based) is used because it captures monotonic parameter-error
        # relationships without assuming linearity, which suits the non-linear model response.
        # Using .dropna() ensures that incomplete optimization runs don't break the calculation
        corr_matrix = df[param_cols + error_cols].dropna().corr(method='spearman')
        sensitivity_matrix = corr_matrix.loc[param_cols, error_cols]

        # 5. Create a Balanced Plot
        # plt.figure(figsize=(6, 8))  # Adjust height to accommodate all parameter rows
        plt.figure()  # Adjust height to accommodate all parameter rows
        ax = plt.gca()

        # Use aspect='auto' to force the matrix to fill the figure width
        im = ax.imshow(sensitivity_matrix, cmap='Greys', vmin=-1, vmax=1, aspect='auto')

        # # Add colorbar with specific padding
        # cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        # cbar.set_label('Spearman Correlation', rotation=270, labelpad=15)

        # Set Axis Labels
        ax.set_xticks(range(len(error_cols)))
        ax.set_xticklabels([format_param_latex(c) for c in error_cols], ha='right')
        ax.set_yticks(range(len(param_cols)))
        ax.set_yticklabels([format_param_latex(c) for c in param_cols])

        # Add numeric values inside the heatmap.
        # In 'Greys' darkness grows with the value (vmin=-1 -> white, vmax=1 -> black),
        # so use white text once the cell is dark enough for contrast.
        for i in range(len(param_cols)):
            for j in range(len(error_cols)):
                val = sensitivity_matrix.iloc[i, j]
                color = 'white' if (val + 1) / 2 > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        plt.tight_layout()

        # Save the finalized figure
        plt.savefig('balanced_sensitivity_beech.png', bbox_inches='tight')
        # 6. Print to terminal with clear labeling
        print("\n" + "="*50)
        print("SENSITIVITY ANALYSIS: SPEARMAN CORRELATION MATRIX")
        print("="*50)

        # We use to_string() to ensure the full table is shown in the terminal
        print(sensitivity_matrix.to_string())

        print("-" * 50)
        print("Top 3 Parameters Increasing Total Error (Bad for Model):")
        top_bad = sensitivity_matrix['error'].sort_values(ascending=False).head(3)
        for param, val in top_bad.items():
            print(f"  -> {param:.<25} Corr: {val:>6.2f}")

        print("\nTop 3 Parameters Reducing Total Error (Good for Model):")
        top_good = sensitivity_matrix['error'].sort_values(ascending=True).head(3)
        for param, val in top_good.items():
            print(f"  -> {param:.<25} Corr: {val:>6.2f}")
        print("="*50)

        plt.show()

