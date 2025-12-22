#!/usr/bin/env python
"""
Plot: Redshift at given SNR threshold vs plunge time Tpl.

This script generates a figure showing the redshift reach for different EMRI sources
at a fixed SNR threshold, plotted against plunge time for fixed primary mass and spin.

Based on cell 22 from degradation_analysis.ipynb (redshift at SNR part)
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
snr_threshold = 30     # SNR threshold for redshift calculation
degradation = 1.0      # Degradation factor (1.0 = no degradation)
m2_filter = 'all'      # Secondary mass filter ('all' or specific value)
estimator = np.median  # Statistical estimator

# -----------------------------------------------------------------------------
# Load detection data
# -----------------------------------------------------------------------------
detection_files = sorted(glob.glob('snr_*/detection.h5'))
print(f"Found {len(detection_files)} detection.h5 files")

source_metadata = {}
source_snr_data = {}

for idx, det_file in enumerate(detection_files):
    source_id = int(det_file.split('_')[1].split('/')[0])
    
    with h5py.File(det_file, 'r') as f:
        source_metadata[source_id] = {
            'm1': float(np.round(f['m1'][()], decimals=5)),
            'm2': float(np.round(f['m2'][()], decimals=5)),
            'a': float(np.round(f['a'][()], decimals=5)),
            'p0': float(f['p0'][()]),
            'e0': float(f['e0'][()]),
            'T': float(np.round(f['Tpl'][()], decimals=5)),
        }
        
        snr_data = f['snr'][()]
        redshifts = f['redshift'][()]
        
        source_snr_data[source_id] = {}
        for z_idx, z_val in enumerate(redshifts):
            source_snr_data[source_id][float(z_val)] = snr_data[z_idx, :]

print(f"Loaded metadata for {len(source_metadata)} sources")

# -----------------------------------------------------------------------------
# Filter sources and compute redshift at SNR threshold
# -----------------------------------------------------------------------------
tolerance = 1e-6
matching_sources = []

for src_idx in sorted(source_metadata.keys()):
    src_a = source_metadata[src_idx]['a']
    src_m1 = source_metadata[src_idx]['m1']
    
    if abs(src_a - spin_a) < tolerance and abs(src_m1 - m1_val) < tolerance:
        matching_sources.append(src_idx)

if not matching_sources:
    raise ValueError(f"No sources found for m1={m1_val:.0e}, a={spin_a:.2f}")

# Extract redshift reach data
z_data = {}
for src_idx in matching_sources:
    tpl = source_metadata[src_idx]['T']
    m2 = source_metadata[src_idx]['m2']
    
    # Get SNR vs redshift
    z_snr_dict = source_snr_data[src_idx]
    z_vals_list = sorted(z_snr_dict.keys())
    snr_data_array = np.array([z_snr_dict[z] for z in z_vals_list])  # shape (n_z, n_realizations)
    
    # Compute median and quantiles for SNR at each z
    snr_median_per_z = np.median(snr_data_array, axis=1)
    snr_lower_per_z = np.quantile(snr_data_array, 0.025, axis=1)
    snr_upper_per_z = np.quantile(snr_data_array, 0.975, axis=1)
    
    z_vals_array = np.array(z_vals_list)
    
    # Check if SNR threshold is achievable
    if snr_threshold > np.max(snr_median_per_z):
        continue
    
    try:
        # Interpolants for original
        interp_median = interp1d(snr_median_per_z, z_vals_array, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
        z_median_orig = interp_median(snr_threshold)
        
        interp_lower = interp1d(snr_lower_per_z, z_vals_array, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
        z_lower_orig = interp_lower(snr_threshold)
        
        interp_upper = interp1d(snr_upper_per_z, z_vals_array, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
        z_upper_orig = interp_upper(snr_threshold)
        
        # For degraded
        snr_median_per_z_deg = snr_median_per_z / np.sqrt(degradation)
        snr_lower_per_z_deg = snr_lower_per_z / np.sqrt(degradation)
        snr_upper_per_z_deg = snr_upper_per_z / np.sqrt(degradation)
        
        interp_median_deg = interp1d(snr_median_per_z_deg, z_vals_array, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
        z_median_deg = interp_median_deg(snr_threshold)
        
        interp_lower_deg = interp1d(snr_lower_per_z_deg, z_vals_array, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        z_lower_deg = interp_lower_deg(snr_threshold)
        
        interp_upper_deg = interp1d(snr_upper_per_z_deg, z_vals_array, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        z_upper_deg = interp_upper_deg(snr_threshold)
        
    except:
        continue
    
    if m2 not in z_data:
        z_data[m2] = {'tpl': [], 'z_median_orig': [], 'z_lower_orig': [], 'z_upper_orig': [],
                      'z_median_deg': [], 'z_lower_deg': [], 'z_upper_deg': []}
    z_data[m2]['tpl'].append(tpl)
    z_data[m2]['z_median_orig'].append(z_median_orig)
    z_data[m2]['z_lower_orig'].append(z_lower_orig)
    z_data[m2]['z_upper_orig'].append(z_upper_orig)
    z_data[m2]['z_median_deg'].append(z_median_deg)
    z_data[m2]['z_lower_deg'].append(z_lower_deg)
    z_data[m2]['z_upper_deg'].append(z_upper_deg)

# Filter out sources with mass ratio q=1e-3 that do not have multiple Tpl runs
q_target = 1e-3
m2_to_remove = []
for m2 in z_data:
    q = m2 / m1_val
    if abs(q - q_target) < 1e-6 and len(z_data[m2]['tpl']) <= 1:
        m2_to_remove.append(m2)

for m2 in m2_to_remove:
    del z_data[m2]

# -----------------------------------------------------------------------------
# Create plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(3.25, 2*2.0))

colors = plt.cm.tab20(np.linspace(0, 1, len(z_data)))

for idx, m2 in enumerate(sorted(z_data.keys())):
    if m2_filter != 'all' and m2 != m2_filter:
        continue
    
    tpl_vals = np.array(z_data[m2]['tpl'])
    z_median_orig = np.array(z_data[m2]['z_median_orig'])
    z_lower_orig = np.array(z_data[m2]['z_lower_orig'])
    z_upper_orig = np.array(z_data[m2]['z_upper_orig'])
    z_median_deg = np.array(z_data[m2]['z_median_deg'])
    z_lower_deg = np.array(z_data[m2]['z_lower_deg'])
    z_upper_deg = np.array(z_data[m2]['z_upper_deg'])
    
    sort_idx = np.argsort(tpl_vals)
    tpl_sorted = tpl_vals[sort_idx]
    z_median_orig_sorted = z_median_orig[sort_idx]
    z_lower_orig_sorted = z_lower_orig[sort_idx]
    z_upper_orig_sorted = z_upper_orig[sort_idx]
    z_median_deg_sorted = z_median_deg[sort_idx]
    z_lower_deg_sorted = z_lower_deg[sort_idx]
    z_upper_deg_sorted = z_upper_deg[sort_idx]
    
    # Error bars: yerr is (lower_error, upper_error) where lower_error = median - lower, upper_error = upper - median
    yerr_orig = [z_median_orig_sorted - z_lower_orig_sorted, z_upper_orig_sorted - z_median_orig_sorted]
    ax.errorbar(tpl_sorted, z_median_orig_sorted, yerr=yerr_orig, fmt='o-', color=colors[idx],
                markersize=7, linewidth=1.5, label=f'{m2:.0f}', alpha=0.7, capsize=3)
    if degradation != 1.0:
        yerr_deg = [z_median_deg_sorted - z_lower_deg_sorted, z_upper_deg_sorted - z_median_deg_sorted]
        ax.errorbar(tpl_sorted, z_median_deg_sorted, yerr=yerr_deg, fmt='s--', color=colors[idx],
                    markersize=6, linewidth=1.5, alpha=0.5, capsize=3)

ax.set_xlabel(r'Plunge time $T_{{pl}} [\mathrm{yr}]$')
ax.set_ylabel(f'Redshift at $\\mathrm{{SNR}}={int(snr_threshold):.0f}$')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', which='major')

# Legend for secondary masses
legend_elements_m2 = [Line2D([0], [0], marker='o', label=f'{m2:.0f}', markersize=7, linestyle='-', color=colors[idx]) 
                      for idx, m2 in enumerate(sorted(z_data.keys())) if (m2_filter == 'all' or m2 == m2_filter)]
leg = ax.legend(handles=legend_elements_m2,
                 bbox_to_anchor=(0.5, 1.02), loc='lower center',
                 frameon=True, ncols=4,
                 title=r'Secondary mass $m_2 [M_\odot]$')

plt.tight_layout()
output_filename = f'z_at_snr_vs_tpl_m1_{m1_val:.0e}_a_{spin_a}.png'
plt.savefig(os.path.join(script_dir, output_filename), dpi=400)
print(f"Plot saved: figures/{output_filename}")
# plt.show()