#!/usr/bin/env python
"""
Plot: Precision metrics vs plunge time Tpl.

This script generates plots showing precision metrics as a function of plunge time
for different secondary masses, at fixed primary mass and spin.

Based on plot_scatter_precision_m1_m2.py for data processing and plot_redshift_at_snr_vs_tpl.py for structure.
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
m1_val = 1e7           # Primary mass [Msun]
spin_a = 0.99          # Spin parameter (prograde)
degradation = 1.0      # Degradation factor (1.0 = no degradation)
selected_run_type = 'circular'
m2_filter = 'all'      # Secondary mass filter ('all' or specific value)

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

# -----------------------------------------------------------------------------
# Filter sources
# -----------------------------------------------------------------------------
tolerance = 1e-6
matching_sources = []

for src_key in sorted(inference_metadata.keys()):
    source_id, run_type = src_key
    src_a = inference_metadata[src_key]['a']
    src_m1 = inference_metadata[src_key]['m1']
    
    if abs(src_a - spin_a) < tolerance and abs(src_m1 - m1_val) < tolerance and run_type == selected_run_type:
        matching_sources.append(src_key)

if not matching_sources:
    raise ValueError(f"No sources found for m1={m1_val:.0e}, a={spin_a:.2f}, run_type={selected_run_type}")

# -----------------------------------------------------------------------------
# Generate plots for each precision metric
# -----------------------------------------------------------------------------
for precision_metric in list(ylabel_map.keys()):
    precision_data = {}
    
    for src_key in matching_sources:
        source_id, run_type = src_key
        tpl = inference_metadata[src_key]['T']
        m2 = inference_metadata[src_key]['m2']
        
        # Check if this precision metric exists for this source
        if precision_metric not in inference_precision_data[src_key]:
            continue
        
        # Get precision array and compute median
        precision_array = inference_precision_data[src_key][precision_metric]
        precision_median = np.median(precision_array)
        precision_deg = precision_median * np.sqrt(degradation)
        
        if m2 not in precision_data:
            precision_data[m2] = {'tpl': [], 'precision': []}
        precision_data[m2]['tpl'].append(tpl)
        precision_data[m2]['precision'].append(precision_deg)

    if not precision_data:
        print(f"No data found for metric {precision_metric}")
        continue
    
    # Filter out m2 with only one Tpl if q=1e-3
    q_target = 1e-3
    m2_to_remove = []
    for m2 in precision_data:
        q = m2 / m1_val
        if abs(q - q_target) < 1e-6 and len(precision_data[m2]['tpl']) <= 1:
            m2_to_remove.append(m2)
    
    for m2 in m2_to_remove:
        del precision_data[m2]
    
    if not precision_data:
        continue

    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2*2.0))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(precision_data)))
    
    for idx, m2 in enumerate(sorted(precision_data.keys())):
        if m2_filter != 'all' and m2 != m2_filter:
            continue
        
        tpl_vals = np.array(precision_data[m2]['tpl'])
        precision_vals = np.array(precision_data[m2]['precision'])
        
        sort_idx = np.argsort(tpl_vals)
        tpl_sorted = tpl_vals[sort_idx]
        precision_sorted = precision_vals[sort_idx]
        
        ax.plot(tpl_sorted, precision_sorted, 'o-', color=colors[idx],
                markersize=7, linewidth=1.5, label=f'{m2:.0f}', alpha=0.7)

    ax.set_xlabel(r'Plunge time $T_{{pl}} [\mathrm{yr}]$')
    ax.set_ylabel(ylabel_map[precision_metric])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major')
    
    # Legend for secondary masses
    legend_elements_m2 = [Line2D([0], [0], marker='o', label=f'{m2:.0f}', markersize=7, linestyle='-', color=colors[idx]) 
                          for idx, m2 in enumerate(sorted(precision_data.keys())) if (m2_filter == 'all' or m2 == m2_filter)]
    leg = ax.legend(handles=legend_elements_m2,
                     bbox_to_anchor=(0.5, 1.02), loc='lower center',
                     frameon=True, ncols=4,
                     title=r'Secondary mass $m_2 [M_\odot]$')

    plt.tight_layout()
    output_filename = f'{precision_metric}_vs_tpl_m1_{m1_val:.0e}_a_{spin_a}.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=400)
    print(f"Plot saved: figures/{output_filename}")
    # plt.show()