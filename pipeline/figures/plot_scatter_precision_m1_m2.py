#!/usr/bin/env python
"""
Plot: Scatter plot of m1 vs m2 color-coded by precision metric.

This script generates scatter plots for all available precision metrics,
showing the primary vs secondary mass space with color-coding indicating
measurement precision levels.

Cell 43 from degradation_analysis.ipynb
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
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
spin_a = 0.99  # Prograde spin
tpl_val = 0.25  # Plunge time
degradation = 1.0
tolerance = 1e-6
selected_run_type = 'circular'

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
    "snr": "SNR",
}

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

# Add SNR to ylabel_map
ylabel_map['snr'] = 'SNR'

# -----------------------------------------------------------------------------
# Generate plots for each precision metric
# -----------------------------------------------------------------------------
for precision_metric in list(ylabel_map.keys()):
    scatter_data = {'m1': [], 'm2': [], 'precision': []}

    for src_key in sorted(inference_metadata.keys()):
        source_id, run_type = src_key
        
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
        
        # Get precision array and compute median (across 100 realizations)
        precision_array = inference_precision_data[src_key][precision_metric]
        precision_median = np.median(precision_array)
        precision_deg = precision_median * np.sqrt(degradation)
        
        # Only add selected run type to avoid duplication
        if run_type == selected_run_type:
            scatter_data['m1'].append(m1)
            scatter_data['m2'].append(m2)
            scatter_data['precision'].append(precision_deg)

    if not scatter_data['m1']:
        print(f"No data found for spin a={spin_a:.2f}, Tpl={tpl_val:.2f} and metric {precision_metric}")
        continue
    
    fig, ax = plt.subplots(figsize=(3.25, 1.5*2.0))
    
    # Create scatter plot
    m1_arr = np.array(scatter_data['m1'])
    m2_arr = np.array(scatter_data['m2'])
    precision_arr = np.abs(np.array(scatter_data['precision']))
    
    # Create scatter plot with color mapping
    prec_min = float(precision_arr.min())
    prec_max = float(precision_arr.max())
    vmin_decade = prec_min * 0.7
    vmax_decade = prec_max * 1.5

    scatter = ax.scatter(m1_arr, m2_arr, c=precision_arr, s=200, cmap='cividis',
                         alpha=0.7, edgecolors='black', linewidth=0.5,
                         norm=plt.matplotlib.colors.LogNorm(vmin=vmin_decade, vmax=vmax_decade))
    
    ax.set_xlabel(r'Primary mass $m_1 [M_\odot]$')
    ax.set_ylabel(r'Secondary mass $m_2 [M_\odot]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 3e4)
    
    # Add colorbar above the plot
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', location='top')
    cbar.set_label(ylabel_map[precision_metric], fontsize=12)
    
    # Set colorbar ticks to the unique precision values
    unique_precision_values = np.sort(np.unique(precision_arr))

    # Add full-width tick lines across the colorbar for min/max values
    cbar_ax = cbar.ax
    for val in [unique_precision_values.min(), unique_precision_values.max()]:
        cbar_ax.axvline(x=val, color='green', linewidth=1.5, alpha=1.0, linestyle='-')
        
    for val in [unique_precision_values.min(), unique_precision_values.max()]:
        cbar_ax.axvline(x=val * 1.4, color='red', linewidth=1.5, alpha=1.0, linestyle='--')

    plt.tight_layout()
    output_filename = f'scatter_{precision_metric}_m1_vs_m2_spin_a{spin_a}.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved: figures/{output_filename}")
    # plt.show()
