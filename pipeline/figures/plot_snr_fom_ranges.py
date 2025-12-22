#!/usr/bin/env python
"""
Plot: SNR vs primary mass m1 with color-coded filled regions for FoM ranges.

This script generates a figure with two subplots (prograde and retrograde spin)
showing SNR at a fixed redshift, with filled regions indicating different 
Figure of Merit (FoM) degradation levels.

Cell 24 from degradation_analysis.ipynb
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os

# Add parent directory to path and change to pipeline directory for data files
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(script_dir)
sys.path.insert(0, pipeline_dir)
os.chdir(pipeline_dir)
from common import CosmoInt

# Use the physrev style if available
try:
    plt.style.use('physrev.mplstyle')
except:
    pass

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
m2_target = 1
Tpl_target = 0.25
z_target_prograde = 0.021544346900318832
z_target_retrograde = 0.021544346900318832  # Change this to use different z for retrograde
tolerance = 1e-6
estimator = np.median

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
# Create figure with two subplots
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(3.25, 2*2.0), sharex=True)

spin_configs = [
    (0.99, 'Prograde $a=+0.99$, $m_2=1M_\\odot$', 'P', z_target_prograde),
    (-0.99, 'Retrograde $a=-0.99$, $m_2=1M_\\odot$', 'X', z_target_retrograde),
]

for ax_idx, (spin_a, spin_label, marker, z_target) in enumerate(spin_configs):
    ax = axes[ax_idx]

    # Find all sources for this spin with matching m2 and Tpl
    matching_sources = []
    for src_idx in sorted(source_metadata.keys()):
        src_a = source_metadata[src_idx]['a']
        src_m2 = source_metadata[src_idx]['m2']
        src_Tpl = source_metadata[src_idx]['T']
        if (abs(src_a - spin_a) < tolerance and
            abs(src_m2 - m2_target) < 0.01 and
            abs(src_Tpl - Tpl_target) < 0.01):
            matching_sources.append(src_idx)

    if not matching_sources:
        print(f"No matching sources found for spin={spin_a}")
        continue

    # Extract SNR data at z
    m1_list = []
    snr_list = []
    for src_idx in matching_sources:
        if z_target not in source_snr_data[src_idx]:
            continue
        m1 = source_metadata[src_idx]['m1']
        snr_array = source_snr_data[src_idx][z_target]
        snr_median = estimator(snr_array)
        m1_list.append(m1)
        snr_list.append(snr_median)

    if not (m1_list and snr_list):
        print(f"No SNR data found for spin={spin_a} at z={z_target}")
        continue

    # Sort by m1
    sort_idx = np.argsort(m1_list)
    m1_sorted = np.array(m1_list)[sort_idx]
    snr_ref = np.array(snr_list)[sort_idx]  # This is d=1 reference

    # Compute SNR at each d level
    snr_d05 = snr_ref / 0.7   # d=0.5 (better SNR)
    snr_d1 = snr_ref          # d=1 (reference)
    snr_d2 = snr_ref / 1.4    # d=2

    # Fill regions between levels
    # Blue: above d=0.5 (from d=0.5 upwards)
    ax.fill_between(m1_sorted, snr_d05, np.ones_like(snr_d05).max() * 500, color='blue', alpha=0.3, label=r'Blue: $d\le0.5$')
    ax.set_ylim(0.0, snr_d05.max() * 1.1)
    
    # Green: from d=0.5 to d=1
    ax.fill_between(m1_sorted, snr_d1, snr_d05, color='green', alpha=0.3, label=r'Green: $0.5<d\le1$')
    
    # Yellow: from d=1 to d=2
    ax.fill_between(m1_sorted, snr_d2, snr_d1, color='goldenrod', alpha=0.3, label=r'Yellow: $1<d\le2$')
    
    # Red: from d=2 downwards
    ax.fill_between(m1_sorted, 0.0, snr_d2, color='red', alpha=0.3, label=r'Red: $d>2$')

    # Plot boundary lines for each level
    ax.plot(m1_sorted, snr_d05, marker=marker, color='blue', markersize=6,
            linewidth=1.5, alpha=0.9, linestyle='-')
    ax.plot(m1_sorted, snr_d1, marker=marker, color='green', markersize=7,
            linewidth=2.0, alpha=0.9, linestyle='-')
    ax.plot(m1_sorted, snr_d2, marker=marker, color='goldenrod', markersize=6,
            linewidth=1.5, alpha=0.9, linestyle='-')

    ax.set_ylabel(rf'SNR at redshift $z={z_target:.4f}$')
    ax.set_title(spin_label, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Legend with FoM range descriptions
range_handles = [
    Patch(facecolor='blue', alpha=0.3, edgecolor='blue', label=r'Blue: $d\le0.5$'),
    Patch(facecolor='green', alpha=0.3, edgecolor='green', label=r'Green: $0.5<d\le1$'),
    Patch(facecolor='goldenrod', alpha=0.3, edgecolor='goldenrod', label=r'Yellow: $1<d\le2$'),
    Patch(facecolor='red', alpha=0.3, edgecolor='red', label=r'Red: $d>2$'),
]
axes[1].legend(handles=range_handles, loc='upper right', frameon=True)

# Shared x-axis label at bottom
axes[-1].set_xlabel(r'Primary mass $m_1 [M_\odot]$')
axes[0].set_xscale('log')
axes[0].set_xlim(5e4, 1e7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'snr_fom_ranges_m2_1_Tpl_0.25_prograde_retrograde.png'), dpi=300, bbox_inches='tight')
print("Plot saved: figures/snr_fom_ranges_m2_1_Tpl_0.25_prograde_retrograde.png")
# plt.show()
