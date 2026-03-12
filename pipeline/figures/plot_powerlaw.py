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
#inference_files_1 = sorted(glob.glob("./../production_inference_m1=1500000.0_m2=150_a=0.99_e_f=0_T=4.5_z=0.5_nr_*/inference.h5"))
inference_files_1 = sorted(glob.glob("./../production_inference_m1=125000.0_m2=12.5_a=0.99_e_f=0_T=0.25_z=0.25_nr_*/inference.h5"))
inference_files_2 = sorted(glob.glob("./../production_inference_m1=125000.0_m2=12.5_a=0.99_e_f=0_T=4.5_z=0.25_nr_*/inference.h5"))
inference_files_3 = sorted(glob.glob("./../production_inference_m1=1500000.0_m2=150_a=0.99_e_f=0_T=4.5_z=0.5_nr_*/inference.h5"))
inference_files =  inference_files_1 + inference_files_2 + inference_files_3 #+ inference_files_1 #+ inference_files_4

print(f"Found {len(inference_files_1)} files for m1=1.25e5, m2=12.5, z=0.25")
print(f"Found {len(inference_files_2)} files for m1=1.25e5, m2=12.5, z=0.25, T=4.5")
print(f"Found {len(inference_files_3)} files for m1=1.5e6, m2=105, z=0.5")
#print(f"Found {len(inference_files_4)} files for m1=1.25e5, m2=12.5, z=0.25, nr=8, dt=4")
print(f"Total: {len(inference_files)} inference.h5 files")

#nr_re = re.compile(r"_nr_(-?\d+)$")  # match at end of folder name
nr_re = re.compile(r"_nr_(-?\d+(?:\.\d+)?)$")  # match integer or float at end of folder name

inference_metadata = {}
inference_precision_data = {}

for inf_file in inference_files:
    parent_dir = os.path.basename(os.path.dirname(inf_file))
    m = nr_re.search(parent_dir)
    if not m:
        raise ValueError(f"Could not parse nr from folder '{parent_dir}' for file: {inf_file}")
    nr = float(m.group(1))

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

        # Key: (m1, m2, z, T, nr)
        T_val = float(np.round(run_group["Tpl"][()], 5))
        source_key = (m1, m2, z, T_val, nr)

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

        detector_precision = run_group["detector_measurement_precision"][()] / 0.82
        source_precision   = run_group["source_measurement_precision"][()] / 0.82
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
A_GW241011 = 3e-3   # GW241011 - quadrupole (n_r = -2)
A_GW230529 = 1.6e-4   # GW230529 - scalar dipole (n_r = 1)

# DM constraint at n_r = 5.5
A_DM_highmass = 1e-5      # rho_DM = 1e17 M_sun/pc^3
A_DM_lowmass = 1e-8       # rho_DM = 1e16 M_sun/pc^3

# Disk constraint at n_r = 8
A_disk_fEdd = 1.5e-6      # f_Edd = 0.01
A_disk_fEdd_low = 1.5e-10 # f_Edd = 0.1

# -----------------------------------------------------------------------------
# Process data - Group by (m1, m2, z) configuration
#   Compute BOTH mean and std dev of sigma_A (abs precision of A)
# -----------------------------------------------------------------------------
config_data = {}
for key in inference_metadata:
    m1, m2, z, T_val, nr = key
    config = (m1, m2, z, T_val)

    if key not in inference_precision_data:
        continue

    snr = np.asarray(inference_metadata[key]["snr"])
    mask = snr > snr_thresh

    if not np.any(mask):
        continue

    if config not in config_data:
        config_data[config] = {'nr': [], 'mean_absA': [], 'std_absA': [],
                               'median_absA': [], 'p16_absA': [], 'p84_absA': []}

    config_data[config]['nr'].append(nr)

    if "absolute_precision_A" in inference_precision_data[key]:
        absA = np.asarray(inference_precision_data[key]["absolute_precision_A"])
        vals = absA[mask]
        config_data[config]['mean_absA'].append(np.mean(vals))
        med = np.median(vals)
        p16, p84 = np.percentile(vals, [16, 84])
        config_data[config]['median_absA'].append(med)
        config_data[config]['p16_absA'].append(p16)
        config_data[config]['p84_absA'].append(p84)
        config_data[config]['std_absA'].append(np.std(vals, ddof=1) if vals.size > 1 else 0.0)
    else:
        config_data[config]['mean_absA'].append(np.nan)
        config_data[config]['median_absA'].append(np.nan)
        config_data[config]['p16_absA'].append(np.nan)
        config_data[config]['p84_absA'].append(np.nan)
        config_data[config]['std_absA'].append(np.nan)

print(f"Found {len(config_data)} configurations:")
for config in sorted(config_data.keys()):
    m1, m2, z, T_val = config
    print(f"  m1={m1:.0e}, m2={m2:.1f}, z={z:.2f}, T={T_val}: {len(config_data[config]['nr'])} nr values")

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
    return rf'${m1_str},\, {m2_str}$'

# -----------------------------------------------------------------------------
# Create color mapping - HARDCODED
# -----------------------------------------------------------------------------
unique_systems = sorted(set((config[0], config[1], config[3]) for config in config_data.keys()))
print(f"\nUnique (m1, m2, T) systems: {len(unique_systems)}")
for s in unique_systems:
    print(f"  {s}")

cmap = plt.cm.tab10

# Assign colors per (m1, m2) mass pair, linestyle by T: 4.5 -> solid, 0.25 -> dashed
# Light system (125000, 12.5) -> orange, Heavy system (1500000, 150) -> blue
mass_pair_colors = {}
for m1, m2, T_val in unique_systems:
    if (m1, m2) not in mass_pair_colors:
        if m1 <= 200000:
            mass_pair_colors[(m1, m2)] = '#ff7f0e'  # orange
        else:
            mass_pair_colors[(m1, m2)] = 'C0'  # blue

styles = {}
for m1, m2, T_val in unique_systems:
    ls = '-' if T_val == 4.5 else '--'
    alpha = 1.0 if T_val == 4.5 else 0.75
    styles[(m1, m2, T_val)] = {'color': mass_pair_colors[(m1, m2)], 'linestyle': ls, 'alpha': alpha}

# -----------------------------------------------------------------------------
# Create plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.75, 3.5))


# -----------------------------------------------------------------------------
# add PRX plot
# -----------------------------------------------------------------------------
include_prx = False
# h5_filename = "all_samples_A_nr_PRX.h5"

# # Load arrays from h5py file
# with h5py.File(h5_filename, "r") as h5f:
#         n10 = h5f["n10"][:]
#         n8 = h5f["n8"][:]
#         n5_9 = h5f["n5_9"][:]
#         n4_4 = h5f["n4_4"][:]
#         n3 = h5f["n3"][:]
#         n2 = h5f["n2"][:]
#         n0 = h5f["n0"][:]
#         n1 = h5f["n1"][:]
#         nm1_2 = h5f["nm1_2"][:]
#         nm2 = h5f["nm2"][:]
#         nm2_5 = h5f["nm2_5"][:]
#         nm3 = h5f["nm3"][:]
#         nm4 = h5f["nm4"][:]
#         nm0_25 = h5f["nm0_25"][:]
#         n0_25 = h5f["n0_25"][:]

# data = [nm4[:362048,-1],nm3[:362048,-1], 
#         nm2_5[:362048,-1],nm2[:362048,-1],nm1_2[:362048,-1],
#         nm0_25[:362048,-1],
#         n0[:362048,-1],
#         n0_25[:362048,-1],
#         n1[:362048,-1], n2[:362048,-1],n3[:362048,-1],  n4_4[:362048,-1], n5_9[:362048,-1], n8[:362048,-1], n10[:362048,-1], ] # wind beta, wind alpha, min beta, mig alpha
# quantile = np.percentile(data, 84, axis=1)
# nr = np.array([-4.0,-3.0, -2.5,-2.0, -1.2, -0.25,0.0, 0.25, 1.0, 2.0, 3.0, 4.4, 5.9, 8.0, 10.0])
# if include_prx:
#     plt.semilogy(nr, quantile, 'o-', ms=2, color='C5')
# plt.fill_between(nr, quantile, y2=1e-7, color='C5', alpha=0.3)
# plt.text(1.5, 1e-6, 'EMRI constraint', color='C5', fontsize=7)
# -----------------------------------------------------------------------------

all_nr_vals = []
all_absA_vals = []

for config, data in sorted(config_data.items()):
    m1, m2, z, T_val = config
    nr_vals     = np.array(data['nr'])
    median_absA = np.array(data['median_absA'])
    p16_absA    = np.array(data['p16_absA'])
    p84_absA    = np.array(data['p84_absA'])

    order = np.argsort(nr_vals)
    nr_vals     = nr_vals[order]
    median_absA = median_absA[order]
    p16_absA    = p16_absA[order]
    p84_absA    = p84_absA[order]

    # Filter to only plot nr >= -2
    in_range = nr_vals >= -2
    nr_vals     = nr_vals[in_range]
    median_absA = median_absA[in_range]
    p16_absA    = p16_absA[in_range]
    p84_absA    = p84_absA[in_range]

    all_nr_vals.extend(nr_vals.tolist())
    all_absA_vals.extend(median_absA.tolist())

    style = styles[(m1, m2, T_val)]

    # Asymmetric error bars: [median - p16, p84 - median]
    yerr_low  = median_absA - p16_absA
    yerr_high = p84_absA - median_absA

    # Line + error bars (median with 16th/84th percentiles)
    ax.errorbar(
        nr_vals, median_absA, yerr=[yerr_low, yerr_high],
        fmt='-',                      # line only, no marker
        color=style['color'],
        linestyle=style['linestyle'],
        linewidth=1,
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        alpha=style['alpha']
    )

# -----------------------------------------------------------------------------
# Set axis limits explicitly
# -----------------------------------------------------------------------------
ax.set_xticks(range(-2, 9))
ax.set_xlim(-2.5, 8.5)
ax.minorticks_off()

ax.set_xlabel(r"Effect power-law $n_r$")
ax.set_ylabel(r"Constraint on effect amplitude $\sigma_A$")
ax.set_yscale("log")

# Set y-limits (include constraint points)
finite_absA = [v for v in all_absA_vals if np.isfinite(v) and v > 0]
y_min = (min(finite_absA) * 0.3) if finite_absA else 1e-20
y_max = max(finite_absA + [A_GW230529, A_GW241011, A_disk_fEdd, A_disk_fEdd_low]) * 3
ax.set_ylim(y_min, 2*y_max)

# -----------------------------------------------------------------------------
# Plot requirement line
# -----------------------------------------------------------------------------
# vals = np.asarray(all_absA_vals, dtype=float)
# requirement = 1.4 * np.max(vals)
# #requirement = 1.4 * np.max(np.stack([np.asarray(all_absA_vals)[:4], np.asarray(all_absA_vals)[4:]]),axis=0)
# ax.plot(nr_vals, requirement, 'r:', lw=1, label='Requirement')
# ax.annotate('Requirement', xy=(nr_vals[len(nr_vals)//2], requirement[len(nr_vals)//2]*2.5),
#             xytext=(nr_vals[len(nr_vals)//2], requirement[len(nr_vals)//2]*2.5),
#             fontsize=7, color='red', ha='center', va='bottom')
# -----------------------------------------------------------------------------
# Set y-axis to show each power of 10
# -----------------------------------------------------------------------------
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
ax.yaxis.set_major_formatter(LogFormatterMathtext())

ax.grid(True, alpha=0.5)

# -----------------------------------------------------------------------------
# Add shaded regions (thin, full y-axis height)
# -----------------------------------------------------------------------------
n_r_quad = -2
n_r_dipole = 1
n_r_disk = 8
n_r_disk_gap = 4
n_r_df = 5.5

ax.axvspan(n_r_disk - region_hw, n_r_disk + region_hw, alpha=0.2, color='purple', zorder=0)
#ax.axvspan(n_r_disk_gap - region_hw, n_r_disk_gap + region_hw, alpha=0.2, color='red', zorder=0)
ax.axvspan(n_r_dipole - region_hw, n_r_dipole + region_hw, alpha=0.2, color='orange', zorder=0)
ax.axvspan(n_r_quad - region_hw, n_r_quad + region_hw, alpha=0.2, color='blue', zorder=0)
ax.axvspan(n_r_df - region_hw, n_r_df + region_hw, alpha=0.2, color='steelblue', zorder=0)

# -----------------------------------------------------------------------------
# Add constraint markers with annotations
# -----------------------------------------------------------------------------
ax.plot(n_r_quad, A_GW241011, '*', color='blue', markersize=10, zorder=10,
        markeredgecolor='white', markeredgewidth=0.3, alpha=0.7)
ax.annotate('GW241011', xy=(n_r_quad, A_GW241011),
            xytext=(n_r_quad + 0.1, A_GW241011 * 1.7),
            fontsize=7, color='black', ha='left', va='bottom')

ax.plot(n_r_dipole, A_GW230529, '*', color='orange', markersize=10, zorder=10,
        markeredgecolor='white', markeredgewidth=0.3, alpha=0.7)
ax.annotate('GW230529', xy=(n_r_dipole, A_GW230529),
            xytext=(n_r_dipole + 0.15, A_GW230529 * 1.4),
            fontsize=7, color='black', ha='left', va='bottom')


# -----------------------------------------------------------------------------
# Susi's points
# -----------------------------------------------------------------------------
# Relativistic model points (Susi's results)
# ax.errorbar([1.15], [5.57e-6], yerr=[[2.01e-6], [3.24e-6]], fmt='x', color='C0', markersize=5, zorder=10, alpha=0.7)
# ax.errorbar([1.15], [1.59e-6], yerr=[[0.56e-6], [0.85e-6]], fmt='x', color='#ff7f0e', markersize=5, zorder=10, alpha=0.7)
ax.scatter([1.15], [4.5e-6], marker='P', color='C0', s=25, zorder=10, alpha=0.7)
ax.scatter([1.15], [3.6e-6], marker='P', color='#ff7f0e', s=25, zorder=10, alpha=0.7)
# arrow to relativistic model
ax.annotate('', xy=(0.7, 0.3e-5), xytext=(-0.5, 1e-6), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('Relativistic\nModel', xy=(-0.5, 1e-7), xytext=(-0.5, 1e-7),
            fontsize=7, color='black', ha='center', va='bottom')
ax.scatter([0.3], [1.6e-7], marker='P', color='k', s=25, zorder=10, alpha=0.5)
# -----------------------------------------------------------------------------
# Create legends
# -----------------------------------------------------------------------------
legend_elements_emri = []
for m1, m2, T_val in unique_systems:
    style = styles[(m1, m2, T_val)]
    T_label = rf'{format_mass_pair(m1, m2)}, {T_val}'
    legend_elements_emri.append(
        Line2D([0], [0],
               label=T_label,
               linestyle=style['linestyle'],
               linewidth=1,
               color=style['color'],
               alpha=1.0)
    )
if include_prx:
    legend_elements_emri.append(Line2D([0], [0], linestyle='-', ms=2, marker='o', color='C5', label=r'$(8 \times 10^5, 40)$'))

# legend_elements_emri.append(Line2D([0], [0], linestyle='-', ms=2, marker='P', color='grey', label=r'Relativistic Model'))

leg1 = ax.legend(handles=legend_elements_emri,
                 loc='lower center', ncols=1,
                 bbox_to_anchor=(0.52, 0.0),
                 title=r'$m_1[M_\odot], m_2[M_\odot], T[\mathrm{yr}]$', frameon=False, framealpha=1.0,
                 fontsize=7, title_fontsize=6)
ax.add_artist(leg1)

# legend_elements_effects = [
#     Patch(facecolor='red', alpha=0.3, edgecolor='red', label=r'Disk Torques'),
#     Patch(facecolor='steelblue', alpha=0.3, edgecolor='steelblue', label=r'Dark Matter'),
#     Patch(facecolor='orange', alpha=0.3, edgecolor='orange', label=r'Scalar charge'),
#     Patch(facecolor='blue', alpha=0.3, edgecolor='blue', label=r'Kerr deviation'),
# ]

# leg2 = ax.legend(handles=legend_elements_effects,
#                  bbox_to_anchor=(1.0, 1.02), loc='lower right', ncols=1,
#                  frameon=False, fontsize=7, title_fontsize=7)

legend_elements_effects = [
    Patch(facecolor='blue', alpha=0.3, edgecolor='blue', label=r'Kerr Deviation'),
    Patch(facecolor='orange', alpha=0.3, edgecolor='orange', label=r'Scalar Charge'),
]
leg1 = ax.legend(handles=legend_elements_effects,
                 bbox_to_anchor=(0.4, 1.00), loc='lower right', ncols=1,
                 frameon=False, fontsize=8, title_fontsize=8, title='Beyond GR effects')
# ax.add_artist(leg1)

legend_elements_effects = [
    Patch(facecolor='steelblue', alpha=0.3, edgecolor='steelblue', label=r'Dark Matter'),
    Patch(facecolor='purple', alpha=0.3, edgecolor='purple', label=r'Accretion Disk'),
]
leg2 = ax.legend(handles=legend_elements_effects,
                 bbox_to_anchor=(1., 1.00), loc='lower right', ncols=1,
                 frameon=False, fontsize=8, title_fontsize=8, title='Environmental effects')
ax.add_artist(leg1)
plt.tight_layout()
plt.savefig(f"nr_amplitude_precision_constraints_prx{include_prx}.png", dpi=300, bbox_inches='tight')
print(f"Plot saved: nr_amplitude_precision_constraints_prx{include_prx}.png")

##############################################################################################################
def get_phi_n(pn_order, m1, m2, M, nu, chi_1, chi_2):
    # from appendix of https://arxiv.org/pdf/2203.13937.pdf
    gamma_E = 0.57721566490153286060  # Euler's constant
    delta = (m1 - m2) / M
    chi_S = (chi_1 + chi_2) / 2
    chi_A = (chi_1 - chi_2) / 2
    if int(2*pn_order) == 2:
        return 3715/756 + 55*nu/9
    elif int(2*pn_order) == 3:
        return -16*np.pi + 113*delta*chi_A/3 + (113/3 - 76*nu/3)*chi_S
    elif int(2*pn_order) == 4:
        return 15293365/508032 + 27145*nu/504 + 3085*nu**2/72 + (-405/8 + 200*nu)*chi_A**2 - 405*delta*chi_A*chi_S/4 + (-405/8 + 5*nu/2)*chi_S**2
    elif int(2*pn_order) == 5:
        return 38645*np.pi/756 - 65*np.pi*nu/9 + (-732985/2268 - 140*nu/9)*delta*chi_A + (-732985/2268 + 24260*nu/81 + 340*nu**2/9)*chi_S
    elif int(2*pn_order) == 6:
        return 11583231236531/4694215680 - 6848*np.log(4)/21 - 640*np.pi**2/3 + 6848*gamma_E/21 + (-15737765635/3048192 + 2255*np.pi**2/12)*nu + 76055*nu**2/1728 - 127825*nu**3/1296 + 2270*np.pi*delta*chi_A/3 + (2270*np.pi/3 - 520*np.pi*nu)*chi_S + (75515/288 - 547945*nu/504 - 8455*nu**2/24)*chi_A**2 + (75515/144 - 8225*nu/18)*delta*chi_A*chi_S + (75515/288 - 126935*nu/252 + 19235*nu**2/72)*chi_S**2
    elif int(2*pn_order) == 7:
        return 77096675*np.pi/254016 + 378515*np.pi*nu/1512 - 74045*np.pi*nu**2/756 + (-25150083775/3048192 + 26804935*nu/6048 - 1985*nu**2/48)*delta*chi_A + (-25150083775/3048192 + 10566655595*nu/762048 - 1042165*nu**2/3024 + 5345*nu**3/36)*chi_S
    else:
        return 1.0

def get_beta_dphi_from_B(B, pn_order, M, mu, a):
    # from eq 21 of https://arxiv.org/pdf/2002.02030
    eta = mu * M / (mu + M)**2 # symmetric mass ratio
    beta = -15/32 * 1/(4-pn_order) * 1/(5-2*pn_order) * B * eta**(-2*pn_order/5) # https://arxiv.org/pdf/1204.2585 also eq 29 of https://arxiv.org/pdf/1603.08955
    # beta(phi_n) eq 10 of https://arxiv.org/pdf/1603.08955
    b = 2*pn_order-5 # power ppE
    delta_phi = 128/3 * beta * eta**(2*pn_order/5) / get_phi_n(pn_order, M, mu, M+mu, eta, a, 0.0) # eq 21 https://arxiv.org/pdf/2002.02030
    return beta, delta_phi


default_width = 3.25 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

fig, axs = plt.subplots(1, 1, figsize=(default_width, default_width * default_ratio*2.0))
for config, data in sorted(config_data.items()):
    m1, m2, z, T_val = config
    nr_vals     = np.array(data['nr'])
    mask = (nr_vals >= -2) & (nr_vals <= 2)
    nr_vals = nr_vals[mask]
    median_absA = np.array(data['median_absA'])[mask]/10**(nr_vals) # convert to A at 1 year before merger
    delta_phi_list = np.abs([get_beta_dphi_from_B(median_absA[ii], -nr_vals[ii], m1*(1+z), m2*(1+z), 0.99)[1] for ii in range(len(nr_vals))])
    style = styles[(m1, m2, T_val)]
    arg_sort = np.argsort(nr_vals)
    
    label_ = rf'{format_mass_pair(m1, m2)}, {T_val}'
    axs.semilogy(-nr_vals[arg_sort], delta_phi_list[arg_sort],'o', alpha=0.8,ms=5, color=style['color'], linestyle=style['linestyle'], label=label_)

dphi_1e6 = np.abs(get_beta_dphi_from_B(4.5e-6 / 10, -1, 1e6, 100, 0.99)[1])
dphi_1e5 = np.abs(get_beta_dphi_from_B(3.6e-6 / 10, -1, 1e5, 10, 0.99)[1])
print(f"dphi for 1e6: {dphi_1e6:.2e}, dphi for 1e5: {dphi_1e5:.2e}")
axs.scatter([-1.15], [dphi_1e6], marker='P', color='C0', s=25, zorder=10, alpha=0.7)
axs.scatter([-1.15], [dphi_1e5], marker='P', color='#ff7f0e', s=25, zorder=10, alpha=0.7)

axs.set_xlabel(r'PN order',fontsize=15)
axs.set_ylabel(r'$|\delta \varphi|$',fontsize=15)


# axs.grid(axis='y')

# LVK bounds from Elise's paper
# axs.semilogy(["-1", "0", "0.5", "1", "1.5", "2"], dphi_lvk,'*',label='LVK')
nr_array = np.array([-1, 0, 0.5, 1, 1.5, 2])
# axs.semilogy(nr_array, [2e-5, 3e-1, 7e-2, 1e-1, 0.25, 3],'*',label='GW170817',alpha=0.5,ms=10)
axs.semilogy(nr_array, np.asarray([8e-5, 5.0, 0.2, 0.2, 0.3, 3]),'*',label='GW230529',alpha=0.5,ms=10, color='red')
axs.semilogy(nr_array, [1e-3, 6e-2, 1.5e-1, 1e-1, 7e-2, 0.4],'D',label='GWTC-3',alpha=0.5,ms=5, color='purple')
axs.semilogy(nr_array, [5.5e-7, 0.230612, 0.0161558, 0.166536, 0.127024, 0.845096],'p',label='ET GW230529',alpha=0.5,ms=10)

indep_ = np.abs(get_beta_dphi_from_B(1.6e-4/10, -1, 1.5, 2.5, 0.0)[1])
axs.semilogy(-1, indep_,'*',label='GW230529',alpha=0.5,ms=10, color='orange')
# independent GW230529 at 
print(f"Independent GW230529 dphi: {indep_:.2e}, from paper {8e-5}")

# ax.plot(-n_r_quad, A_GW241011, '*', color='blue', markersize=10, zorder=10,
#         markeredgecolor='white', markeredgewidth=0.3, alpha=0.7)
# ax.plot(-n_r_dipole, A_GW230529, '*', color='orange', markersize=10, zorder=10,
#         markeredgecolor='white', markeredgewidth=0.3, alpha=0.7)


# {{-1, 5.54476*10^-7}, {-(1/2), 0.0000212187}, {0, 0.230612}, {1/2, 0.0161558}, {1, 0.166536}, {3/2, 0.127024}, {2, 0.845096}}
# pulsar bounds # # https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
beta2 = 4e-6
Pdot_precision = 1.3e-4
boundB_prx_dpsr = 4e-10 # 95% quoted in the paper, 
# if we divide by two we obtain one sigma
B = np.random.normal(0,boundB_prx_dpsr / 2,size=10000)
beta_dp, dphi_dp = get_beta_dphi_from_B(B, -1, 1.33818, 1.24886, 0.0)
nr_array = [float(el) for el in ["-1", "0", "0.5", "1"]]
# axs.semilogy(nr_array, [np.quantile(dphi_dp,0.95), 0.8e-4, 0.8, 10.0],'X',label='PTA J0737–3039',alpha=0.5,ms=8, color='green')
# axs.set_ylim(0.5e-11,30)
axs.yaxis.set_major_locator(LogLocator(base=10,numticks=30))  # Set the number of y-axis ticks

legend_elements_effects = [
    Line2D([0], [0], marker='D', color='w', label='GWTC-3', markerfacecolor='purple', markersize=5, alpha=0.7),
    Line2D([0], [0], marker='*', color='w', label='GW230529', markerfacecolor='red', markersize=10, alpha=0.7),
    Line2D([0], [0], marker='*', color='w', label='approximate \n mapping \n GW230529', markerfacecolor='orange', markersize=10, alpha=0.7),
    Line2D([0], [0], marker='p', color='w', label='ET GW230529', markerfacecolor='C0', markersize=10, alpha=0.7),
    # Line2D([0], [0], marker='X', color='w', label='PTA J0737–3039', markerfacecolor='green', markersize=8, alpha=0.7),
]
leg2 = axs.legend(handles=legend_elements_effects,
                  loc='lower right',
                 ncols=1, title='Ground-based constraints',
                 frameon=False, fontsize=7, title_fontsize=7)
axs.add_artist(leg2)

leg1 = axs.legend(handles=legend_elements_emri,
                 loc='upper left', ncols=1,
                 bbox_to_anchor=(0.0, 0.99),
                 title=r'EMRI constraints \\ $m_1[M_\odot], m_2[M_\odot], T[\mathrm{yr}]$', frameon=False, framealpha=1.0,
                 fontsize=7, title_fontsize=6)
axs.add_artist(leg1)
axs.set_ylim(0.002e-7, 10)
plt.savefig(f'./bound_delta_phi.pdf', bbox_inches='tight')  
