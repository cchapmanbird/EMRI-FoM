#!/usr/bin/env python
"""
Polar plot: Measurement precision for different EMRI sources.

This script generates polar plots showing the measurement precision for each
parameter, with radius as precision and angle based on primary mass m1.
Different secondary masses m2 are shown as different colored arcs, mirroring
the redshift-horizon polar plots.

Data sourced from inference results (same as scatter precision plots).
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
from math import log10, floor

np.random.seed(42)

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

# Enable LaTeX rendering for bold text
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
spin_a = 0.99           # Prograde spin
tpl_val = 0.25          # Plunge time
degradation = 1.0       # Degradation factor (1.0 = no degradation)
tolerance = 1e-6
selected_run_type = 'circular'

# Label mapping for precision metrics
ylabel_map = {
    "relative_precision_m1": r"$\sigma_{m_{1}}/m_{1}$",
    "relative_precision_m2": r"$\sigma_{m_{2}}/m_{2}$",
    "relative_precision_dist": r"$\sigma_{d_L}/d_L$",
    "relative_precision_a": r"$\sigma_{a}/a$",
    "absolute_precision_a": r"$\sigma_{a}$",
    "absolute_precision_OmegaS": r"Sky Localization $\Delta\Omega_S\;[\mathrm{deg}^2]$",
    "snr": "SNR",
}

# Title mapping for a cleaner figure title
title_map = {
    "relative_precision_m1": r"Primary Mass Precision $\sigma_{m_1}/m_1$",
    "relative_precision_m2": r"Secondary Mass Precision $\sigma_{m_2}/m_2$",
    "relative_precision_dist": r"Distance Precision $\sigma_{d_L}/d_L$",
    "relative_precision_a": r"Spin Precision $\sigma_{a}/a$",
    "absolute_precision_a": r"Spin Precision $\sigma_{a}$",
    "absolute_precision_OmegaS": r"Sky Localization $\Delta\Omega_S\;[\mathrm{deg}^2]$",
    "snr": r"Signal-to-Noise Ratio",
}

# -----------------------------------------------------------------------------
# Load inference data (same logic as plot_scatter_precision_m1_m2.py)
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
# Helper: format with significant figures
# -----------------------------------------------------------------------------
def format_sigfigs(v, n=2):
    """Format value with n significant figures, avoiding scientific notation when possible."""
    if v == 0:
        return '0'
    magnitude = floor(log10(abs(v)))
    str_out = f'{v/(10**(magnitude)):.0f}' + fr'$\times 10^{{{magnitude}}}$'
    if v / (10**(magnitude)) == 1.0:
        str_out = rf'$10^{{{magnitude}}}$'
    return str_out

# -----------------------------------------------------------------------------
# Collect data grouped by m2 for each precision metric
# -----------------------------------------------------------------------------
for precision_metric in list(ylabel_map.keys()):
    # Group precision data by m2, then by m1
    precision_by_m2 = {}
    all_m1_vals = set()

    for src_key in sorted(inference_metadata.keys()):
        source_id, run_type = src_key
        
        if run_type != selected_run_type:
            continue
        
        # Filter by spin and Tpl
        src_a = inference_metadata[src_key]['a']
        src_tpl = inference_metadata[src_key]['T']
        if abs(src_a - spin_a) > tolerance or abs(src_tpl - tpl_val) > tolerance:
            continue
        
        m1 = inference_metadata[src_key]['m1']
        m2 = inference_metadata[src_key]['m2']
        
        # Check if this precision metric exists for this source
        if precision_metric not in inference_precision_data[src_key]:
            continue
        
        # Get precision array and compute median (across realizations)
        precision_array = inference_precision_data[src_key][precision_metric]
        precision_median = np.median(np.abs(precision_array))
        precision_deg = precision_median * np.sqrt(degradation)
        
        all_m1_vals.add(m1)
        
        if m2 not in precision_by_m2:
            precision_by_m2[m2] = {'m1': [], 'precision': []}
        precision_by_m2[m2]['m1'].append(m1)
        precision_by_m2[m2]['precision'].append(precision_deg)

    if not precision_by_m2:
        print(f"No data for metric {precision_metric} — skipping.")
        continue

    # Sort m1 values to define angular sectors
    m1_list = sorted(all_m1_vals)
    num_m1 = len(m1_list)
    
    if num_m1 == 0:
        continue

    # Map each m1 to an angular sector
    m1_to_sector = {}
    for i, m1_val in enumerate(m1_list):
        theta_min = i * (2 * np.pi / num_m1)
        theta_max = (i + 1) * (2 * np.pi / num_m1)
        m1_to_sector[m1_val] = (theta_min, theta_max)

    # -------------------------------------------------------------------------
    # Create polar plot
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(3.25 * 2.0, 2.0 * 2.))
    ax = fig.add_subplot(111, polar=True)
    
    # Assign colors to m2 values
    m2_list = sorted(precision_by_m2.keys())
    num_m2 = len(m2_list)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_m2, 1)))
    m2_color_dict = {m2: colors[idx] for idx, m2 in enumerate(m2_list)}

    theta_rot = -np.pi / (2 * num_m1)  # Small rotation for aesthetics

    for idx, m2 in enumerate(m2_list):
        m1_vals = np.array(precision_by_m2[m2]['m1'])
        prec_vals = np.array(precision_by_m2[m2]['precision'])
        
        for i, m1_val in enumerate(m1_vals):
            if m1_val not in m1_to_sector:
                continue
            theta_min, theta_max = m1_to_sector[m1_val]
            theta = np.linspace(theta_min, theta_max, 100)
            ax.plot(theta + theta_rot, prec_vals[i] * np.ones_like(theta),
                    color=m2_color_dict[m2], markersize=5, alpha=0.7,
                    linestyle='-', linewidth=2)

    # Set log scale for radial axis (precision spans orders of magnitude)
    ax.set_rscale('log')
    ax.set_rlabel_position(90)

    # Radial grid ticks: auto-determine from data
    all_prec = []
    for m2 in precision_by_m2:
        all_prec.extend(precision_by_m2[m2]['precision'])
    all_prec = np.array(all_prec)
    prec_min_log = np.floor(np.log10(all_prec.min()))
    prec_max_log = np.ceil(np.log10(all_prec.max()))
    r_ticks = [10**e for e in np.arange(prec_min_log, prec_max_log + 1)]
    # Filter ticks within data range
    r_ticks = [r for r in r_ticks if r >= all_prec.min() * 0.5 and r <= all_prec.max() * 2]
    if len(r_ticks) > 0:
        def _fmt_rtick(r):
            """Format a radial tick value as valid LaTeX."""
            if r == 0:
                return r'$0$'
            exp = int(np.floor(np.log10(abs(r))))
            coeff = r / 10**exp
            if abs(coeff - 1.0) < 1e-9:
                return rf'$10^{{{exp}}}$'
            return rf'${coeff:.0f}\times10^{{{exp}}}$'
        ax.set_rgrids(r_ticks, angle=88,
                      labels=[_fmt_rtick(r) for r in r_ticks])

    # Rotate radial labels to vertical
    for label in ax.get_yticklabels():
        label.set_rotation(90)

    ax.grid(True, alpha=0.8, linewidth=2)
    ax.spines['polar'].set_visible(False)

    # Set theta ticks at sector midpoints with m1 labels
    theta_grids_deg = [(i + 0.5) * (360 / num_m1) + np.rad2deg(theta_rot) for i in range(num_m1)]
    ax.set_thetagrids(theta_grids_deg, labels=[''] * num_m1)

    # Add custom m1 labels
    r_label = ax.get_ylim()[1] * 1.3
    for i, m1_val in enumerate(m1_list):
        angle_rad = np.deg2rad(theta_grids_deg[i])
        label_text = format_sigfigs(m1_val)
        ax.text(angle_rad, r_label, label_text, ha='center', va='center',
                fontsize=11, rotation=(angle_rad % np.pi) * 180 / np.pi - 90,
                rotation_mode='anchor')

    # Title
    metric_label = title_map.get(precision_metric, precision_metric)
    ax.text(np.pi / 2, ax.get_ylim()[1] * 2.0, metric_label,
            ha='center', va='bottom', fontsize=12)

    # Legend for secondary masses
    legend_elements_m2 = [
        Line2D([0], [0], marker=None, label=format_sigfigs(m2),
               linestyle='-', linewidth=2, color=m2_color_dict[m2])
        for m2 in m2_list
    ]
    ax.legend(handles=legend_elements_m2, bbox_to_anchor=(0.5, -0.15),
              loc='upper center', ncols=min(num_m2, 4),
              title=r'Secondary mass $m_2\;[M_\odot]$')

    plt.subplots_adjust(bottom=0.18)
    output_filename = f'precision_polar_{precision_metric}_a_{spin_a}_tpl_{tpl_val}.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Polar precision plot saved: figures/{output_filename}")

print("\nAll polar precision plots generated.")
