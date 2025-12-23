#!/usr/bin/env python
"""
Polar plot: Redshift horizon for different sources.

This script generates a polar plot showing the redshift reach for different EMRI sources
at a fixed SNR threshold (30), with radius as redshift and angle based on secondary mass m2.

Based on plot_redshift_at_snr.py.
"""

import h5py
import numpy as np
import glob
import json
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
from common import CosmoInt

# Use the physrev style if available
try:
    plt.style.use('physrev.mplstyle')
except:
    pass

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
tpl_val = 0.25          # Plunge time
spin_a = 0.99           # Spin parameter (prograde)
snr_threshold = 30      # SNR threshold for redshift calculation
degradation = 1.0       # Degradation factor (1.0 = no degradation)
m2_filter = 'all'       # Secondary mass filter ('all' or specific value)
estimator = np.median   # Statistical estimator

# Load LVK GW events
with open('lvk_gw_events.json', 'r') as f:
    lvk_events = json.load(f)

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
    src_tpl = source_metadata[src_idx]['T']
    
    if abs(src_a - spin_a) < tolerance and abs(src_tpl - tpl_val) < tolerance:
        matching_sources.append(src_idx)

if not matching_sources:
    raise ValueError(f"No sources found for Tpl={tpl_val:.2f}, a={spin_a:.2f}")

# Extract redshift reach data
z_data = {}
all_m1 = []
for src_idx in matching_sources:
    m1 = source_metadata[src_idx]['m1']
    m2 = source_metadata[src_idx]['m2']
    all_m1.append(m1)
    
    z_snr_dict = source_snr_data[src_idx]
    z_vals_list = sorted(z_snr_dict.keys())
    snr_median_per_z = []
    
    for z in z_vals_list:
        snr_array = z_snr_dict[z]
        snr_median_per_z.append(np.median(snr_array))
    
    snr_median_per_z = np.array(snr_median_per_z)
    z_vals_array = np.array(z_vals_list)
    
    if snr_threshold > np.max(snr_median_per_z):
        continue
    
    try:
        interp_func = interp1d(snr_median_per_z, z_vals_array, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        z_at_snr = interp_func(snr_threshold)
        
        snr_median_per_z_deg = snr_median_per_z / np.sqrt(degradation)
        interp_func_deg = interp1d(snr_median_per_z_deg, z_vals_array, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        z_at_snr_deg = interp_func_deg(snr_threshold)
    except:
        continue
    
    if m2 not in z_data:
        z_data[m2] = {'m1': [], 'z_orig': [], 'z_deg': []}
    z_data[m2]['m1'].append(m1)
    z_data[m2]['z_orig'].append(z_at_snr)
    z_data[m2]['z_deg'].append(z_at_snr_deg)

# -----------------------------------------------------------------------------
# Create polar plot
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(3.25*2, 2.0*2))
ax = fig.add_subplot(111, polar=True)

if all_m1:
    m1_min = np.log10(min(all_m1))
    m1_max = np.log10(max(all_m1))
    m1_range = m1_max - m1_min

colors = plt.cm.tab20(np.linspace(0, 1, len(z_data)))

m1_color_dict = {5e4: 'C0', 1e5: 'C1', 1e6: 'C2', 1e7: 'C3'}

m2_list = sorted(z_data.keys())
num_m2 = len(m2_list)
theta_rot = -np.pi/12
for idx, m2 in enumerate(m2_list):
    theta_min = idx * (2 * np.pi / num_m2)
    theta_max = (idx + 1) * (2 * np.pi / num_m2)
    
    m1_vals = np.array(z_data[m2]['m1'])
    z_orig = np.array(z_data[m2]['z_orig'])
    for i, m1_val in enumerate(m1_vals):
        color = m1_color_dict[m1_val]
        theta = np.linspace(theta_min, theta_max, 100) 
        plt.plot(theta + theta_rot, z_orig[i]*np.ones_like(theta),  color=color, markersize=5, alpha=0.7, linestyle='-', linewidth=2)

# Add LVK events
theta_lvk = []
r_lvk = []
for m2_lvk, z_lvk in zip(lvk_events['secondary_mass'], lvk_events['redshift']):
    diffs = [abs(m2_lvk - m2) for m2 in m2_list]
    closest_idx = np.argmin(diffs)
    m2_closest = m2_list[closest_idx]
    idx = m2_list.index(m2_closest)
    theta_min = idx * (2 * np.pi / num_m2)
    theta_max = (idx + 1) * (2 * np.pi / num_m2)
    theta = np.random.uniform(theta_min, theta_max)
    theta_lvk.append(theta)
    r_lvk.append(z_lvk)

ax.plot(theta_lvk + np.asarray(theta_rot), r_lvk, marker='.', color='blue', markersize=2, alpha=0.1, linestyle='None', label='LVK GW events')

# Add LVK events
theta_lvk = []
r_lvk = []
for m2_lvk, z_lvk in zip(lvk_events['primary_mass'], lvk_events['redshift']):
    diffs = [abs(m2_lvk - m2) for m2 in m2_list]
    closest_idx = np.argmin(diffs)
    m2_closest = m2_list[closest_idx]
    idx = m2_list.index(m2_closest)
    theta_min = idx * (2 * np.pi / num_m2)
    theta_max = (idx + 1) * (2 * np.pi / num_m2)
    theta = np.random.uniform(theta_min, theta_max)
    theta_lvk.append(theta)
    r_lvk.append(z_lvk)

ax.plot(theta_lvk + np.asarray(theta_rot), r_lvk, marker='.', color='blue', markersize=3, alpha=0.1, linestyle='None', label='LVK GW events')

# # Legend for LVK
leg_lvk = ax.legend([Line2D([0], [0], marker='.', color='blue', markersize=3,alpha=0.1, linestyle='None')], ['LVK GW events'], bbox_to_anchor=(0.9, 0.0), loc='center', frameon=True)
ax.add_artist(leg_lvk)

ax.set_rlabel_position(90)
ax.text(np.pi/2, ax.get_ylim()[1] * 1.1, 'Redshift', ha='center', va='bottom')

ax.set_rscale('log')

# Set radial ticks for redshift
r_ticks = [0.02, 0.2, 2]
ax.set_rgrids(r_ticks, labels=[rf'${r:g}$' for r in r_ticks])

# Rotate radial labels to vertical
for label in ax.get_yticklabels():
    label.set_rotation(90)

ax.grid(True, alpha=0.8, linewidth=2)

ax.spines['polar'].set_visible(False)

# Set theta ticks
theta_grids = [(idx + 0.5) * (360 / num_m2) + theta_rot * 180 / np.pi for idx in range(num_m2)]
labels = [rf'${m2:.0f}M_\odot$' for m2 in m2_list]
labels[-1] = r'$10^{4} M_\odot$'
labels[-2] = r'$10^{3} M_\odot$'
ax.set_thetagrids(theta_grids, labels=labels)

# Legend
from math import log10, floor
def format_sigfigs(v, n=2):
    """Format value with n significant figures, avoiding scientific notation when possible."""
    if v == 0:
        return '0'
    magnitude = floor(log10(abs(v)))
    str_out = f'{v/(10**(magnitude)):.0f}' + rf'$\times 10^{{{magnitude}}}$'
    if v/(10**(magnitude)) == 1.0:
        str_out = rf'$10^{{{magnitude}}}$'
    return str_out

legend_elements_m1 = [Line2D([0], [0], marker=None, label=format_sigfigs(m1), markersize=7, linestyle='-', linewidth=2, color=m1_color_dict[m1]) 
                      for m1 in sorted(m1_color_dict.keys())]
ax.legend(handles=legend_elements_m1, bbox_to_anchor=(0.5, 1.05), loc='lower center', ncols=4, title=r'Primary mass $m_1 [M_\odot]$')

plt.tight_layout()
output_filename = f'redshift_horizon_polar_m2_a_{spin_a}_tpl_{tpl_val}.png'
plt.savefig(os.path.join(script_dir, output_filename), dpi=400)
print(f"Polar plot saved: figures/{output_filename}")