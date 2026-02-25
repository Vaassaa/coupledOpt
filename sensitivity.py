import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data, skipping the comment line
df = pd.read_csv('de_log_analysis.csv', comment='#')
df.columns = df.columns.str.strip()

# 2. Linearize the log10 parameters (alpha_org, alpha_min, K_min, K_org, S_max)
# Note: Spearman correlation remains the same after this, but it's physically correct for your analysis
log_params = ['alpha_org[1/m]', 'alpha_min[1/m]', 'K_min[m/s]', 'K_org[m/s]', 'S_max[m/s]']
for col in log_params:
    if col in df.columns:
        df[col] = 10**df[col]

# 3. Define target columns
error_cols = ['error', 'error_heat', 'error_moist']
exclude_cols = ['call_id', 'timestamp'] + error_cols
param_cols = [c for c in df.columns if c not in exclude_cols]

# 4. Compute Spearman correlation
# Using .dropna() ensures that incomplete optimization runs don't break the calculation
corr_matrix = df[param_cols + error_cols].dropna().corr(method='spearman')
sensitivity_matrix = corr_matrix.loc[param_cols, error_cols]

# 5. Create a Balanced Plot
plt.figure(figsize=(8, 10))  # Adjust height to accommodate all parameter rows
ax = plt.gca()

# Use aspect='auto' to force the matrix to fill the figure width
im = ax.imshow(sensitivity_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

# Add colorbar with specific padding
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Spearman Correlation', rotation=270, labelpad=15)

# Set Axis Labels
ax.set_xticks(range(len(error_cols)))
ax.set_xticklabels(error_cols, rotation=45, ha='right')
ax.set_yticks(range(len(param_cols)))
ax.set_yticklabels(param_cols)

# Add numeric values inside the heatmap
for i in range(len(param_cols)):
    for j in range(len(error_cols)):
        val = sensitivity_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

plt.title('Sensitivity Analysis: Parameters vs. Errors', pad=20)
plt.tight_layout()

# Save the finalized figure
plt.savefig('balanced_sensitivity_heatmap2.png', bbox_inches='tight')
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
