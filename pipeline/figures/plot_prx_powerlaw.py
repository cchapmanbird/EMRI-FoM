import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import json
import os
import h5py
import sys
import glob
import re
from math import log10, floor

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
pipeline_dir = os.path.abspath('./..')  # Go up one level from figures/
sys.path.insert(0, pipeline_dir)
try:
    plt.style.use(os.path.join(pipeline_dir, 'physrev.mplstyle'))
    print("Loaded physrev.mplstyle")
except Exception:
    print("physrev.mplstyle not found, using default style")

# -----------------------------------------------------------------------------
# Load all three inference file sets and combine them
# -----------------------------------------------------------------------------
inference_files_1 = sorted(glob.glob("./../production_inference_m1=1500000.0_m2=75_a=0.99_e_f=0_T=4.5_z=0.5_*/inference.h5"))
inference_files_2 = sorted(glob.glob("./../production_inference_m1=125000.0_m2=12.5_a=0.99_e_f=0_T=4.5_z=0.25_*/inference.h5"))
inference_files_3 = sorted(glob.glob("./../production_inference_m1=1500000.0_m2=150_a=0.99_e_f=0_T=4.5_z=0.5_*/inference.h5"))
inference_files = inference_files_1 + inference_files_2 + inference_files_3

print(f"Found {len(inference_files_1)} files for m1=1.5e6, m2=75, z=0.5")
print(f"Found {len(inference_files_2)} files for m1=1.25e5, m2=12.5, z=0.25")
print(f"Found {len(inference_files_3)} files for m1=1.5e5, m2=150, z=0.5")
print(f"Total: {len(inference_files)} inference.h5 files")

nr_re = re.compile(r"_nr_(-?\d+)$")  # match at end of folder name

inference_metadata = {}
inference_precision_data = {}

for inf_file in inference_files:
    parent_dir = os.path.basename(os.path.dirname(inf_file))
    m = nr_re.search(parent_dir)
    if not m:
        raise ValueError(f"Could not parse nr from folder '{parent_dir}' for file: {inf_file}")
    nr = int(m.group(1))

    with h5py.File(inf_file, "r") as f:
        run_type = "circular"
        if run_type not in f:
            print(f"Warning: '{run_type}' not found in {inf_file}. Available groups: {list(f.keys())}")
            continue

        run_group = f[run_type]

        # Read values used for the key
        m1 = float(np.round(run_group["m1"][()], 5))
        m2 = float(np.round(run_group["m2"][()], 5))
        z  = float(run_group["redshift"][()])

        # Key: (m1, m2, z, nr)
        source_key = (m1, m2, z, nr)

        # Metadata
        inference_metadata[source_key] = {
            "nr": nr,
            "m1": m1,
            "m2": m2,
            "a": float(run_group["a"][()]),
            "p0": float(run_group["p0"][()]),
            "e0": float(run_group["e0"][()]),
            "e_f": float(run_group["e_f"][()]),
            "dist": float(run_group["dist"][()]),
            "T": float(np.round(run_group["Tpl"][()], 5)),
            "redshift": z,
            "snr": run_group["snr"][()],
            "run_type": run_type,
            "fish_params": run_group["fish_params"][()],
        }

        detector_precision = run_group["detector_measurement_precision"][()]
        source_precision   = run_group["source_measurement_precision"][()]
        param_names = np.array(run_group["param_names"][()], dtype=str).tolist()
        inference_metadata[source_key]["param_names"] = param_names

        inference_precision_data[source_key] = {}
        name_to_i = {n: i for i, n in enumerate(param_names)}

        for name, i in name_to_i.items():
            if name == "M":
                inference_precision_data[source_key]["relative_precision_m1_det"] = (
                    detector_precision[:, i] / (m1 * (1 + z))
                )
                inference_precision_data[source_key]["relative_precision_m1"] = (
                    source_precision[:, i] / m1
                )

            elif name == "mu":
                inference_precision_data[source_key]["relative_precision_m2_det"] = (
                    detector_precision[:, i] / (m2 * (1 + z))
                )
                inference_precision_data[source_key]["relative_precision_m2"] = (
                    source_precision[:, i] / m2
                )

            elif name == "e0":
                e0 = inference_metadata[source_key]["e0"]
                inference_precision_data[source_key]["relative_precision_e0"] = (
                    detector_precision[:, i] / e0 if e0 != 0 else np.full(detector_precision.shape[0], np.nan)
                )

            else:
                inference_precision_data[source_key]["absolute_precision_" + name] = detector_precision[:, i]

            if name in ("dist", "a"):
                denom = inference_metadata[source_key][name]
                inference_precision_data[source_key]["relative_precision_" + name] = (
                    detector_precision[:, i] / denom if denom != 0 else np.full(detector_precision.shape[0], np.nan)
                )

print("\nData loading complete!")
print(f"Loaded metadata for {len(inference_metadata)} runs")
print(f"Loaded precision data for {len(inference_precision_data)} runs")

# Unique values (now pulled from the keys)
m1_values = sorted({k[0] for k in inference_metadata})
m2_values = sorted({k[1] for k in inference_metadata})
z_values  = sorted({k[2] for k in inference_metadata})
nr_values = sorted({k[3] for k in inference_metadata})
print("m1:", m1_values)
print("m2:", m2_values)
print("z:", z_values)
print("nr:", nr_values)

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
snr_thresh = 30.0

# Shaded region half-width
region_hw = 0.03

# LVK constraint values
A_GW250114 = 6e-3   # GW250114 - quadrupole (n_r = -2)
A_GW230529 = 6.4e-5   # GW230529 - scalar dipole (n_r = 1)

# Disk constraint at n_r = 8
A_disk_fEdd = 1.5e-6      # f_Edd = 0.01
A_disk_fEdd_low = 1.5e-10 # f_Edd = 0.1

# -----------------------------------------------------------------------------
# Process data - Group by (m1, m2, z) configuration
#   Compute BOTH mean and std dev of sigma_A (abs precision of A)
# -----------------------------------------------------------------------------
config_data = {}
for key in inference_metadata:
    m1, m2, z, nr = key
    config = (m1, m2, z)

    if key not in inference_precision_data:
        continue

    snr = np.asarray(inference_metadata[key]["snr"])
    mask = snr > snr_thresh

    if not np.any(mask):
        continue

    if config not in config_data:
        config_data[config] = {'nr': [], 'mean_absA': [], 'std_absA': []}

    config_data[config]['nr'].append(nr)

    if "absolute_precision_A" in inference_precision_data[key]:
        absA = np.asarray(inference_precision_data[key]["absolute_precision_A"])
        vals = absA[mask]
        config_data[config]['mean_absA'].append(np.mean(vals))
        config_data[config]['std_absA'].append(np.std(vals, ddof=1) if vals.size > 1 else 0.0)
    else:
        config_data[config]['mean_absA'].append(np.nan)
        config_data[config]['std_absA'].append(np.nan)

print(f"Found {len(config_data)} configurations:")
for config in sorted(config_data.keys()):
    m1, m2, z = config
    print(f"  m1={m1:.0e}, m2={m2:.1f}, z={z:.2f}: {len(config_data[config]['nr'])} nr values")

# -----------------------------------------------------------------------------
# Helper function for formatting
# -----------------------------------------------------------------------------
def format_smart(v):
    """Format value: use scientific notation only for pure powers of 10 >= 100."""
    if v == 0:
        return '0'
    magnitude = floor(log10(abs(v)))
    coeff = v / (10**magnitude)
    if coeff == 1.0 and v >= 100:
        if magnitude == 1:
            return '10'
        else:
            return rf'10^{{{magnitude}}}'
    else:
        if v == int(v):
            return rf'{int(v)}'
        else:
            return rf'{v}'

def format_mass_pair(m1, m2):
    """Format (m1, m2) pair for legend."""
    m1_str = format_smart(m1)
    m2_str = format_smart(m2)
    return rf'$({m1_str},\, {m2_str})$'

# -----------------------------------------------------------------------------
# Create color mapping - HARDCODED
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Create plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.25/1.2, 2*1.7/1.2))

# -----------------------------------------------------------------------------
# Define the 4 effects with their n_r values and colors
# -----------------------------------------------------------------------------
effects = {
    'Kerr\nDeviation': {'nr': -2.0, 'color': 'blue', 'lvk_constraint': A_GW250114, 'lvk_label': 'GW250114'},
    'Scalar\nCharge': {'nr': 1.0, 'color': 'orange', 'lvk_constraint': A_GW230529, 'lvk_label': 'GW230529'},
    'Dark\nMatter': {'nr': 5.9, 'color': 'red', 'lvk_constraint': None, 'lvk_label': None},
    'Accretion\nDisk': {'nr': 8.0, 'color': 'purple', 'lvk_constraint': None, 'lvk_label': None},
}

effect_names = list(effects.keys())
x_positions = np.arange(len(effect_names))

# -----------------------------------------------------------------------------
# Load PRX data and extract values for the 4 effects
# -----------------------------------------------------------------------------
h5_filename = "all_samples_A_nr_PRX.h5"

# Load arrays from h5py file
with h5py.File(h5_filename, "r") as h5f:
        n8 = h5f["n8"][:]
        n5_9 = h5f["n5_9"][:]
        n1 = h5f["n1"][:]
        nm2 = h5f["nm2"][:]

# Map n_r values to their data arrays
nr_to_data = {
    -2.0: nm2[:362048, -1],
    1.0: n1[:362048, -1],
    5.9: n5_9[:362048, -1],
    8.0: n8[:362048, -1],
}

# Calculate 84th percentile for each effect
emri_constraints = []
for name in effect_names:
    nr_val = effects[name]['nr']
    data = nr_to_data[nr_val]
    quantile = np.percentile(data, 84)
    emri_constraints.append(quantile)

# -----------------------------------------------------------------------------
# Plot EMRI constraints as points
# -----------------------------------------------------------------------------
colors = [effects[name]['color'] for name in effect_names]
ax.scatter(x_positions, emri_constraints, marker='o', s=50, c=colors, zorder=10, 
           edgecolors='white', linewidths=0.5, label='EMRI constraint')

# Add fill below EMRI constraints
for i, (x, y, color) in enumerate(zip(x_positions, emri_constraints, colors)):
    ax.bar(x, y, width=0.6, bottom=1e-7, color=color, alpha=0.3, zorder=1)

# -----------------------------------------------------------------------------
# Add LVK constraint arrows for Kerr Deviation and Scalar Charge
# -----------------------------------------------------------------------------
for i, name in enumerate(effect_names):
    if effects[name]['lvk_constraint'] is not None:
        lvk_val = effects[name]['lvk_constraint']
        color = effects[name]['color']
        label = effects[name]['lvk_label']
        
        # Add downward triangle marker for LVK constraint
        ax.scatter(x_positions[i], lvk_val, marker='v', s=50, c=color, zorder=10,
                   edgecolors='white', linewidths=0.5)
        
        ax.annotate(label, xy=(x_positions[i]-0.35, lvk_val*1.4),
                    xytext=(x_positions[i]-0.35, lvk_val*1.4),
                    fontsize=7, color='black', ha='left', va='bottom')

# -----------------------------------------------------------------------------
# Configure axes
# -----------------------------------------------------------------------------
ax.set_xticks(x_positions)
ax.set_xticklabels(effect_names, fontsize=8)
# Color each x-tick label with the corresponding effect color
for i, (label, name) in enumerate(zip(ax.get_xticklabels(), effect_names)):
    label.set_color(effects[name]['color'])
ax.set_xlim(-0.5, len(effect_names) - 0.5)

ax.set_ylabel(r"Deviation from GW emission")
ax.set_yscale("log")

# -----------------------------------------------------------------------------
# Add category labels below the effect names
# -----------------------------------------------------------------------------
# Get the axis transform for positioning text below x-axis
trans = ax.get_xaxis_transform()

# "GR modifications" centered below Kerr Deviation and Scalar Charge (positions 0 and 1)
ax.text(0.5, -0.2, "General Relativity \n Modifications", transform=trans, ha='center', va='top', 
        fontsize=8, fontstyle='italic')

# "Environmental Effects" centered below Dark Matter and Accretion Disk (positions 2 and 3)
ax.text(2.5, -0.2, "Environmental \n Effects", transform=trans, ha='center', va='top', 
        fontsize=8, fontstyle='italic')

# Add subtle separating line between the two categories
# ax.axvline(x=1.5, color='grey', linestyle='--', alpha=0.3, lw=0.8)

# Set y-limits
ax.set_ylim(5e-7, 2e-2)

# -----------------------------------------------------------------------------
# Set y-axis to show each power of 10
# -----------------------------------------------------------------------------
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
ax.yaxis.set_major_formatter(LogFormatterMathtext())

# ax.grid(True, alpha=0.5, axis='y')

# -----------------------------------------------------------------------------
# Add legend
# -----------------------------------------------------------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='LISA constraint'),
    Line2D([0], [0], marker='v', color='grey', markersize=8, linestyle='None', label='Ground-based \n current constraint'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig("nr_amplitude_precision_constraints.png", dpi=300, bbox_inches='tight')
print("Plot saved: nr_amplitude_precision_constraints.png")