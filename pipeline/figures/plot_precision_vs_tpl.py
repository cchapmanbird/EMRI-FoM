#!/usr/bin/env python
"""
Plot: Precision metrics vs plunge time Tpl.

This script generates plots showing precision metrics as a function of plunge time
for different secondary masses, at fixed primary mass and spin.

Based on plot_scatter_precision_m1_m2.py for data processing and plot_redshift_at_snr_vs_tpl.py for structure.
"""

import h5py
from matplotlib.patches import Patch
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

# ...existing code...
# -----------------------------------------------------------------------------
# Generate plots for each precision metric
# -----------------------------------------------------------------------------
improvement_data = {}  # To store improvements for the final plot

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
        # print improvements compared to tpl[0]
        # The above code is a Python code snippet with comments. It seems like the code is intended
        # for improvement or further development, but without the actual code inside the comments, it
        # is not possible to determine what specific improvements or changes are being suggested.
        improvement = precision_sorted[1:] / precision_sorted[0]
        print(f"m1={m1_val:.0e} m2={m2:.0f}: {improvement}")
        
        # Store improvement for final plot
        if precision_metric not in improvement_data:
            improvement_data[precision_metric] = {}
        improvement_data[precision_metric][m2] = {
            '1.5': improvement[0],
            '4.5': improvement[1]
        }

    ax.set_xlabel(r'Plunge time $T_{{pl}} [\mathrm{yr}]$')
    ax.set_ylabel(ylabel_map[precision_metric])
    if precision_metric != 'relative_precision_dist':
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

# -----------------------------------------------------------------------------
# Final plot: Improvement vs Parameters (Precision Metrics)
# -----------------------------------------------------------------------------
# Label mapping for precision metrics
# ylabel_map = {
#     "relative_precision_m1_det": r"$\frac{\sigma_{m_{ 1,\mathrm{det} } }}{m_{ 1,\mathrm{det} }}$",
#     "relative_precision_m1": r"$\frac{\sigma_{m_{1} }}{m_{1}}$",
#     "relative_precision_m2_det": r"$\frac{\sigma_{m_{ 2,\mathrm{det} } }}{m_{ 2,\mathrm{det} }}$",
#     "relative_precision_m2": r"$\frac{\sigma_{m_{2} }}{m_{2}}$",
#     "relative_precision_dist": r"$\frac{\sigma_{d_L}}{d_L}$",
#     "relative_precision_e0": r"$\frac{\sigma_{e_0}}{e_0}$",
#     "absolute_precision_a": r"$\sigma_{a}$",
#     "relative_precision_a": r"$\frac{\sigma_{a}}{a}$",
#     "absolute_precision_OmegaS": r"$\Delta \Omega_S$",
#     "snr": "SNR",
# }

# Select only specific metrics
selected_metrics = [
    "relative_precision_m1_det",
    "relative_precision_m2_det",
    "relative_precision_a",
    # "relative_precision_dist",
    "absolute_precision_OmegaS"
]

# Filter improvement_data to only selected metrics
filtered_improvement_data = {metric: improvement_data[metric] for metric in selected_metrics if metric in improvement_data}

if not filtered_improvement_data:
    print("No data for selected metrics.")
else:
    fig, ax = plt.subplots(1, 1, figsize=(3.25*2, 2.0*1.5))
    
    # Get all unique m2 from filtered data
    all_m2 = set()
    for metric in filtered_improvement_data:
        all_m2.update(filtered_improvement_data[metric].keys())
    all_m2 = sorted(all_m2)
    
    # Colors for m2
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_m2)))
    m2_to_color = {m2: colors[i] for i, m2 in enumerate(all_m2)}
    
    # Bar positions
    metrics = list(filtered_improvement_data.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(all_m2)
    
    # Collect handles for legend
    legend_handles = []
    
    for i, m2 in enumerate(all_m2):
        x_pos = x + i * width
        imp_1_5 = [filtered_improvement_data[metric].get(m2, {}).get('1.5', 1.0) for metric in metrics]
        imp_4_5 = [filtered_improvement_data[metric].get(m2, {}).get('4.5', 1.0) for metric in metrics]
        
        # Bottom bar for 1.5 yr
        print(imp_1_5)
        bar1 = ax.bar(x_pos, imp_1_5, width, color=m2_to_color[m2], alpha=0.7)
        # Top bar for additional to 4.5 yr
        bar2 = ax.bar(x_pos, np.array(imp_4_5) - np.array(imp_1_5), width, bottom=imp_1_5, color=m2_to_color[m2], alpha=0.5, hatch='//')
    
    # Create legend handles for m2
    for m2 in all_m2:
        legend_handles.append(Line2D([0], [0], color=m2_to_color[m2], linewidth=4, label=rf'$m_2={m2:.0f} M_\odot$'))
    
    legend_handles.append(Patch(facecolor='grey', alpha=0.5, hatch='//', label=r'$T_{\rm pl} = 1.5$ yr'))
    legend_handles.append(Patch(facecolor='grey', alpha=0.5, label=r'$T_{\rm pl} = 4.5$ yr'))

    
    ax.set_xlabel('Precision Metric')
    ax.set_ylabel(r'(Metric Ratio at $T_{\rm pl}$ / $T_{\rm pl}=0.25$ yr)')
    ax.set_xticks(x + width * (len(all_m2) - 1) / 2)
    ax.set_xticklabels([ylabel_map.get(m, m) for m in metrics], rotation=0, ha='right')
    ax.legend(handles=legend_handles, ncols=6, bbox_to_anchor=(0.5, 1.05), loc='lower center', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.0)
    plt.tight_layout()
    output_filename = f'improvement_vs_parameters_m1_{m1_val:.0e}_a_{spin_a}.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=400)
    print(f"Improvement plot saved: figures/{output_filename}")

# -----------------------------------------------------------------------------
# Additional plot: Relative Precision of m1_det, m2_det, and a vs Tpl
# -----------------------------------------------------------------------------

# Select metrics including dist
selected_metrics = [
    "relative_precision_m1_det",
    "relative_precision_m2_det",
    "relative_precision_a",
    "absolute_precision_OmegaS",
    "relative_precision_dist"
]

# Collect data for the selected metrics
precision_vs_tpl_data = {}
for metric in selected_metrics:
    precision_vs_tpl_data[metric] = {}
    for src_key in matching_sources:
        source_id, run_type = src_key
        tpl = inference_metadata[src_key]['T']
        m2 = inference_metadata[src_key]['m2']
        
        if metric not in inference_precision_data[src_key]:
            continue
        
        precision_array = inference_precision_data[src_key][metric]
        precision_median = np.median(precision_array) * np.sqrt(degradation)
        
        if m2 not in precision_vs_tpl_data[metric]:
            precision_vs_tpl_data[metric][m2] = {'tpl': [], 'precision': []}
        precision_vs_tpl_data[metric][m2]['tpl'].append(tpl)
        precision_vs_tpl_data[metric][m2]['precision'].append(precision_median)

# Collect redshift data
redshift_vs_tpl_data = {}
for src_key in matching_sources:
    source_id, run_type = src_key
    tpl = inference_metadata[src_key]['T']
    m2 = inference_metadata[src_key]['m2']
    redshift = inference_metadata[src_key]['redshift']
    
    if m2 not in redshift_vs_tpl_data:
        redshift_vs_tpl_data[m2] = {'tpl': [], 'redshift': []}
    redshift_vs_tpl_data[m2]['tpl'].append(tpl)
    redshift_vs_tpl_data[m2]['redshift'].append(redshift)

# Define combined metrics
combined_metrics = ["relative_precision_m1_det", "relative_precision_m2_det"]#, "relative_precision_a"]
linestyles = ['-', '--', ':']

# Create the plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(3.25, 4*2.0), sharex=True)

axes = [ax1, ax2, ax3, ax4]
metric_names = ["redshift", "combined", "absolute_precision_OmegaS", "relative_precision_dist"]

# Colors for m2
all_m2 = set()
for metric in precision_vs_tpl_data:
    all_m2.update(precision_vs_tpl_data[metric].keys())
all_m2.update(redshift_vs_tpl_data.keys())
all_m2 = sorted(all_m2)
colors = plt.cm.tab20(np.linspace(0, 1, len(all_m2)))
m2_to_color = {m2: colors[i] for i, m2 in enumerate(all_m2)}

for i, (ax, metric) in enumerate(zip(axes, metric_names)):
    if metric == "redshift":
        for m2 in sorted(redshift_vs_tpl_data.keys()):
            if m2_filter != 'all' and m2 != m2_filter:
                continue
            
            tpl_vals = np.array(redshift_vs_tpl_data[m2]['tpl'])
            redshift_vals = np.array(redshift_vs_tpl_data[m2]['redshift'])
            
            sort_idx = np.argsort(tpl_vals)
            tpl_sorted = tpl_vals[sort_idx]
            redshift_sorted = redshift_vals[sort_idx]
            
            ax.plot(tpl_sorted, redshift_sorted, 'o-', color=m2_to_color[m2],
                    markersize=5, linewidth=1.5, label=f'{m2:.0f}', alpha=0.8)
        
        ax.set_ylabel(r"Redshift at $\mathrm{SNR}=30$")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.legend(title=r'Secondary mass $m_2 [M_\odot]$', bbox_to_anchor=(0.5, 1.02), loc='lower center', ncols=5)
    elif metric == "combined":
        for m2 in sorted(all_m2):
            if m2_filter != 'all' and m2 != m2_filter:
                continue
            
            for j, met in enumerate(combined_metrics):
                if met not in precision_vs_tpl_data or m2 not in precision_vs_tpl_data[met]:
                    continue
                
                tpl_vals = np.array(precision_vs_tpl_data[met][m2]['tpl'])
                precision_vals = np.array(precision_vs_tpl_data[met][m2]['precision'])
                
                sort_idx = np.argsort(tpl_vals)
                tpl_sorted = tpl_vals[sort_idx]
                precision_sorted = precision_vals[sort_idx]
                if met == "relative_precision_m1_det":
                    markersize = 6
                else:
                    markersize = 3
                ax.plot(tpl_sorted, precision_sorted, linestyle=linestyles[j], marker='o', color=m2_to_color[m2],
                        markersize=markersize, linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel(r"Relative Precision")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Legend for metrics
        
        legend_elements = [Line2D([0], [0], linestyle=ls, marker='o',markersize=6 if met == "relative_precision_m1_det" else 3, color='black', label=ylabel_map[met]) for ls, met in zip(linestyles, combined_metrics)]
        ax.legend(handles=legend_elements, loc='upper right', ncols=1, frameon=True)
    else:
        for m2 in sorted(precision_vs_tpl_data[metric].keys()):
            if m2_filter != 'all' and m2 != m2_filter:
                continue
            
            tpl_vals = np.array(precision_vs_tpl_data[metric][m2]['tpl'])
            precision_vals = np.array(precision_vs_tpl_data[metric][m2]['precision'])
            
            sort_idx = np.argsort(tpl_vals)
            tpl_sorted = tpl_vals[sort_idx]
            precision_sorted = precision_vals[sort_idx]
            
            ax.plot(tpl_sorted, precision_sorted, 'o-', color=m2_to_color[m2],
                    markersize=5, linewidth=1.5, label=f'{m2:.0f}', alpha=0.8)
        
        ax.set_ylabel(ylabel_map[metric])
        ax.grid(True, alpha=0.3)
        if metric != 'relative_precision_dist':
            ax.set_yscale('log')

ax4.set_xlabel(r'Mission Duration [yr]')

plt.tight_layout()
output_filename = f'redshift_combined_precision_vs_tpl_m1_{m1_val:.0e}_a_{spin_a}.png'
plt.savefig(os.path.join(script_dir, output_filename), dpi=400)
print(f"Additional plot saved: figures/{output_filename}")

# -----------------------------------------------------------------------------
# ...existing code...