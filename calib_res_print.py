import numpy as np
from datetime import datetime
# ---- parameters from calibration ----
params = np.loadtxt('calib_res/best_guess_spruce.in')

# ---- names + units in the same order ----
param_info = [
    ("b1_org", "W/(m·K)"),
    ("b2_org", "W/(m·K)"),
    ("b3_org", "W/(m·K)"),
    ("b1_min", "W/(m·K)"),
    ("b2_min", "W/(m·K)"),
    ("b3_min", "W/(m·K)"),
    ("albedo", "-"),
    ("alpha_org", "1/m"),
    ("n_org", "-"),
    ("K_org", "m/s"),
    ("alpha_min", "1/m"),
    ("n_min", "-"),
    ("K_min", "m/s"),
    ("S_max", "m/s"),
]

# ---- generate markdown ----
now = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
lines.append("# Calibration Results\n")
lines.append("| Parameter | Value | Unit |")
lines.append("|-----------|------:|------|")

for (name, unit), val in zip(param_info, params):
    lines.append(f"| {name} | {val:.6g} | {unit} |")

# ---- write file ----
with open("arch/spruce_calib_res.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("calib_res.md written.")

