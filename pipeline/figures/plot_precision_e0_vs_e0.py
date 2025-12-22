#!/usr/bin/env python
"""
Plot: Relative precision on eccentricity vs initial eccentricity e0.

This script generates a figure showing the measurement precision on initial
eccentricity as a function of eccentricity itself, color-coded by mass ratio
and with different markers for each primary mass.

Cell 46 from degradation_analysis.ipynb
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import log10, floor
import sys
import os

# Add parent directory to path and change to pipeline directory for data files
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(script_dir)
sys.path.insert(0, pipeline_dir)
os.chdir(pipeline_dir)

# Use the physrev style if available
try:
    plt.style.use('physrev.mplstyle')
except:
    pass

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
spin_target = 0.99
tpl_target = 0.25
run_type_target = 'eccentric'
precision_metric = 'relative_precision_e0'
tolerance = 1e-6
estimator = np.median

# Label mapping for precision metrics
ylabel_map = {
    "relative_precision_m1_det": r"$\sigma_{m_{ 1,\mathrm{det} } }/m_{ 1,\mathrm{det} }$",
    "relative_precision_m1": r"$\sigma_{m_{1} }/m_{1}$",
    "relative_precision_m2_det": r"$\sigma_{m_{ 2,\mathrm{det} } }/m_{ 2,\mathrm{det} }$",
    "relative_precision_m2": r"$\sigma_{m_{2} }/m_{2}$",
    "relative_precision_dist": r"$\sigma_{d_L}/d_L$",
    "relative_precision_e0": r"$\sigma_{e_0}/e_0$",
    "absolute_precision_a": r"$\sigma_{a}$",
    "relative_precision_a": r"$\sigma_{a}/a$",
    "absolute_precision_OmegaS": r"$\Delta \Omega_S$",
}

# -----------------------------------------------------------------------------
# Helper function
# -----------------------------------------------------------------------------
def format_sigfigs(v, n=2):
    """Format value with n significant figures, avoiding scientific notation when possible."""
    if v == 0:
        return '0'
    magnitude = floor(log10(abs(v)))
    str_out = f'{v/(10**(magnitude)):.0f}' + rf'$\times 10^{{{magnitude}}}$'
    if v/(10**(magnitude)) == 1.0:
        str_out = rf'$10^{{{magnitude}}}$'
    return str_out

# -----------------------------------------------------------------------------
# Load inference data
# -----------------------------------------------------------------------------
inference_files = sorted(glob.glob('inference_*/inference.h5'))
print(f"Found {len(inference_files)} inference.h5 files")

inference_metadata = {}
inference_precision_data = {}

for idx, inf_file in enumerate(inference_files):
    source_id = int(inf_file.split('_')[1].split('/')[0])
    
    with h5py.File(inf_file, 'r') as f:
        for run_type in ['circular', 'eccentric']:
            if run_type not in f.keys():
                continue
            
            run_group = f[run_type]
            source_key = (source_id, run_type)
            
            inference_metadata[source_key] = {
                'm1': float(np.round(run_group['m1'][()], decimals=5)),
                'm2': float(np.round(run_group['m2'][()], decimals=5)),
                'a': float(run_group['a'][()]),
                'p0': float(run_group['p0'][()]),
                'e0': float(run_group['e0'][()]),
                'e_f': float(run_group['e_f'][()]),
                'dist': float(run_group['dist'][()]),
                'T': float(np.round(run_group['Tpl'][()], decimals=5)),
                'redshift': float(run_group['redshift'][()]),
                'snr': run_group['snr'][()],
                'run_type': run_type,
            }
            
            detector_precision = run_group['detector_measurement_precision'][()]
            source_precision = run_group['source_measurement_precision'][()]
            param_names = run_group['param_names'][()]
            param_names = np.array(param_names, dtype=str).tolist()
            inference_metadata[source_key].update({"param_names": param_names})

            for ii, name in enumerate(param_names):
                if name == 'M':
                    if source_key not in inference_precision_data:
                        inference_precision_data[source_key] = {}
                    inference_precision_data[source_key].update({
                        "relative_precision_m1_det": detector_precision[:, param_names.index(name)] / (inference_metadata[source_key]['m1'] * (1 + inference_metadata[source_key]['redshift'])),
                        "relative_precision_m1": source_precision[:, param_names.index(name)] / inference_metadata[source_key]['m1']
                    })
                elif name == 'mu':
                    if source_key not in inference_precision_data:
                        inference_precision_data[source_key] = {}
                    inference_precision_data[source_key].update({
                        "relative_precision_m2_det": detector_precision[:, param_names.index(name)] / (inference_metadata[source_key]['m2'] * (1 + inference_metadata[source_key]['redshift'])),
                        "relative_precision_m2": source_precision[:, param_names.index(name)] / inference_metadata[source_key]['m2']
                    })
                elif name == 'e0':
                    if source_key not in inference_precision_data:
                        inference_precision_data[source_key] = {}
                    inference_precision_data[source_key].update({
                        "relative_precision_e0": detector_precision[:, param_names.index(name)] / inference_metadata[source_key]['e0']
                    })
                else:
                    if source_key not in inference_precision_data:
                        inference_precision_data[source_key] = {}
                    inference_precision_data[source_key].update({
                        "absolute_precision_" + name: detector_precision[:, param_names.index(name)]
                    })
                
                if name == 'dist':
                    if source_key not in inference_precision_data:
                        inference_precision_data[source_key] = {}
                    inference_precision_data[source_key].update({
                        "relative_precision_" + name: detector_precision[:, param_names.index(name)] / inference_metadata[source_key][name]
                    })
                if name == 'a':
                    if source_key not in inference_precision_data:
                        inference_precision_data[source_key] = {}
                    inference_precision_data[source_key].update({
                        "relative_precision_" + name: detector_precision[:, param_names.index(name)] / inference_metadata[source_key][name]
                    })
            
            inference_precision_data[source_key].update({"snr": run_group['snr'][()]})

print(f"Loaded metadata for {len(inference_metadata)} sources")

# -----------------------------------------------------------------------------
# Collect data grouped by (mass ratio q, m1)
# -----------------------------------------------------------------------------
data_by_q_m1 = {}

for src_key in sorted(inference_metadata.keys()):
    source_id, run_type = src_key
    if run_type != run_type_target:
        continue

    meta = inference_metadata[src_key]
    if abs(meta['a'] - spin_target) > tolerance:
        continue
    if abs(meta['T'] - tpl_target) > tolerance:
        continue

    if precision_metric not in inference_precision_data[src_key]:
        continue

    m1 = meta['m1']
    m2 = meta['m2']
    e0 = meta['e0']
    q = m2 / m1  # mass ratio
    
    precision_array = np.asarray(inference_precision_data[src_key][precision_metric], dtype=float)
    precision_val = estimator(precision_array)
    p2p5, p97p5 = np.percentile(precision_array, [2.5, 97.5])

    # Round q for grouping (to handle floating point)
    q_key = np.round(q, decimals=10)
    key = (q_key, m1)
    if key not in data_by_q_m1:
        data_by_q_m1[key] = {'e0': [], 'prec': [], 'p2p5': [], 'p97p5': [], 'm1': m1, 'm2': m2, 'q': q_key}
    
    data_by_q_m1[key]['e0'].append(e0)
    data_by_q_m1[key]['prec'].append(precision_val)
    data_by_q_m1[key]['p2p5'].append(p2p5)
    data_by_q_m1[key]['p97p5'].append(p97p5)

if len(data_by_q_m1) == 0:
    raise ValueError(
        f"No data found for a={spin_target:.2f}, Tpl={tpl_target}, "
        f"run_type={run_type_target}, metric={precision_metric}"
    )

# -----------------------------------------------------------------------------
# Create plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(3.25, 1.5*2.0))

# Get unique mass ratios and primary masses
q_keys = sorted(set(k[0] for k in data_by_q_m1.keys()))
m1_keys = sorted(set(k[1] for k in data_by_q_m1.keys()))

# Filter out mass ratio 2e-5
q_keys = [q for q in q_keys if not np.isclose(q, 2e-5, rtol=1e-2)]

# Create colormap for mass ratios
colors = plt.cm.tab10(np.linspace(0, 1, len(q_keys)))
q_to_color = {q: colors[idx] for idx, q in enumerate(q_keys)}

# Define markers for each m1 value
marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
m1_to_marker = {m1: marker_list[idx % len(marker_list)] for idx, m1 in enumerate(m1_keys)}

# Collect all data points grouped by mass ratio q for drawing connecting lines
data_by_q = {}
for (q_key, m1), data in data_by_q_m1.items():
    if q_key not in data_by_q:
        data_by_q[q_key] = {'e0': [], 'prec': []}
    data_by_q[q_key]['e0'].extend(data['e0'])
    data_by_q[q_key]['prec'].extend(data['prec'])

# Draw connecting lines for each mass ratio (for legend)
for q_key in q_keys:
    e0_all = np.asarray(data_by_q[q_key]['e0'])
    prec_all = np.asarray(data_by_q[q_key]['prec'])
    sort_idx = np.argsort(e0_all)
    e0_sorted = e0_all[sort_idx]
    prec_sorted = prec_all[sort_idx]
    
    color = q_to_color[q_key]
    ax.plot(e0_sorted, prec_sorted, '-', color=color, linewidth=1.5, alpha=0.6,
            label=format_sigfigs(q_key))

# Plot the data points with markers
for (q_key, m1), data in sorted(data_by_q_m1.items()):
    e0_arr = np.asarray(data['e0'])
    prec_arr = np.asarray(data['prec'])
    p2p5_arr = np.asarray(data['p2p5'])
    p97p5_arr = np.asarray(data['p97p5'])

    sort_idx = np.argsort(e0_arr)
    e0_arr = e0_arr[sort_idx]
    prec_arr = prec_arr[sort_idx]
    p2p5_arr = p2p5_arr[sort_idx]
    p97p5_arr = p97p5_arr[sort_idx]

    # Asymmetric errorbars
    yerr_lower = np.maximum(0.0, prec_arr - p2p5_arr)
    yerr_upper = np.maximum(0.0, p97p5_arr - prec_arr)
    yerr = np.vstack([yerr_lower, yerr_upper])
    
    if q_key not in q_to_color:
        continue
    color = q_to_color[q_key]
    marker = m1_to_marker[m1]
    
    ax.errorbar(
        e0_arr, prec_arr,
        fmt=marker, linestyle='None', color=color, markersize=6,
        elinewidth=1.0, capsize=2, alpha=0.8
    )

ax.set_xlabel(r'Initial eccentricity $e_0$')
ax.set_ylabel(ylabel_map.get(precision_metric, precision_metric))
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
# ax.set_title(rf'$a={spin_target:.2f},\; T_{{pl}}={tpl_target}$ yr')

# First legend for mass ratio (colors) - uses the lines
leg1 = ax.legend(frameon=False, loc='upper right', ncol=1, title=r'$m_2/m_1$')
ax.add_artist(leg1)

# Second legend for primary mass (markers)
marker_handles = [Line2D([0], [0], marker=m1_to_marker[m1], color='gray', linestyle='None', markersize=6, label=format_sigfigs(m1)) for m1 in m1_keys]
leg2 = ax.legend(handles=marker_handles, frameon=False, loc='lower left', ncol=2, title=r'Primary mass $m_1 \, [M_\odot]$', framealpha=0.0)
ax.add_artist(leg2)
ax.set_xlim(None, 1)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'precision_e0_vs_e0_by_mass_ratio_and_m1.png'), dpi=300)
print("Plot saved: figures/precision_e0_vs_e0_by_mass_ratio_and_m1.png")
# plt.show()
