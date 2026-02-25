import numpy as np
from datetime import datetime

# ---- parameters from calibration ----
pars = np.array([
    0.19245167,
    0.87277299,
    0.072251397,
    0.64501215,
    0.85465909,
    3.205033,
    0.050442908,
    0.69260894,
    2.0948597,
    -4.2603004,
    0.26173936,
    1.3463667,
    -3.7358467,
    1.7952128e-08
])
pars = np.array([
    0.25657747,
    0.67764785,
    0.090741717,
    0.97103077,
    0.76294462,
    3.192622,
    0.042552,
    0.6593947,
    2.186195,
    -3.7887338,
    0.23160688,
    1.332452,
    -4.0271087,
    4.416331e-08
])

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

# ---- indices that must be converted from log10 space ----
log10_indices = {
    "alpha_org",
    "K_org",
    "alpha_min",
    "K_min",
}

# ---- apply conversion ----
converted = []
for (name, unit), val in zip(param_info, pars):
    if name in log10_indices:
        val = 10 ** val
    converted.append(val)

converted = np.array(converted)

# ---- generate markdown ----
now = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
lines.append("# Calibration Results\n")
lines.append("| Parameter | Value | Unit |")
lines.append("|-----------|------:|------|")

for (name, unit), val in zip(param_info, converted):
    lines.append(f"| {name} | {val:.6g} | {unit} |")

# ---- write file ----
with open("calib_res.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("calib_res.md written.")

