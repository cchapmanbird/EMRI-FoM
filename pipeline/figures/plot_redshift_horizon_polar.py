#!/usr/bin/env python
"""
Polar plot: Redshift horizon for different sources.

This script generates a polar plot showing the redshift reach for different EMRI sources
at a fixed SNR threshold (30), with radius as redshift and angle based on primary mass.

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

# -----------------------------------------------------------------------------
# Load electromagnetic observation data
# -----------------------------------------------------------------------------
# QPE and QPO data (https://arxiv.org/pdf/2404.00941)
masses_qpe = np.asarray([1.2, 0.55, 0.55, 3.1, 42.5, 1.8, 5.5, 0.595, 6.55, 88.0, 5.8]) * 1e6
z_qpe = np.asarray([0.0181, 0.0505, 0.0175, 0.024, 0.044, 0.0237, 0.042, 0.13, 0.0206, 0.0136, 0.0053])

# AGN data from Table EM_measure arXiv-2501.03252v2
smbh_data = [
    {"name": "UGC 01032", "mass": 1.1, "redshift": 0.01678, "alternate_names": "Mrk 359"},
    {"name": "UGC 12163", "mass": 1.1, "redshift": 0.02468, "alternate_names": "Ark 564"},
    {"name": "Swift J2127.4+5654", "mass": 1.5, "redshift": 0.01400, "alternate_names": ""},
    {"name": "NGC 4253", "mass": 1.8, "redshift": 0.01293, "alternate_names": "UGC 07344, Mrk 766"},
    {"name": "NGC 4051", "mass": 1.91, "redshift": 0.00234, "alternate_names": "UGC 07030"},
    {"name": "NGC 1365", "mass": 2.0, "redshift": 0.00545, "alternate_names": ""},
    {"name": "1H0707-495", "mass": 2.3, "redshift": 0.04056, "alternate_names": ""},
    {"name": "MCG-6-30-15", "mass": 2.9, "redshift": 0.00749, "alternate_names": ""},
    {"name": "NGC 5506", "mass": 5.0, "redshift": 0.00608, "alternate_names": "Mrk 1376"},
    {"name": "IRAS13224-3809", "mass": 6.3, "redshift": 0.06579, "alternate_names": ""},
    {"name": "Ton S180", "mass": 8.1, "redshift": 0.06198, "alternate_names": ""},
]
smbh_masses = np.array([item['mass'] for item in smbh_data]) * 1e6
smbh_redshifts = np.array([item['redshift'] for item in smbh_data])

# SDSS DR16Q Quasars
with h5py.File('sdss_dr16q_quasars.h5', 'r') as f:
    redshift_sdss = f['redshift'][:]
    log10massbh_sdss = f['log10massbh'][:]
    log10massbh_err_sdss = f['log10massbh_err'][:]
    relative_error_mass = log10massbh_err_sdss * np.log(10)
    mask = (log10massbh_sdss < 7.05) & (relative_error_mass < 0.5)
    redshift_sdss = redshift_sdss[mask]
    log10massbh_sdss = log10massbh_sdss[mask]
    massbh_sdss = 10**log10massbh_sdss

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

for idx, m2 in enumerate(sorted(z_data.keys())):
    if m2_filter != 'all' and m2 != m2_filter:
        continue
    
    m1_vals = np.array(z_data[m2]['m1'])
    z_orig = np.array(z_data[m2]['z_orig'])
    for i, m1_val in enumerate(m1_vals):
        # Convert m1 to angle
        if np.abs(m1_val - 5e4)<1e-6:
            theta_min, theta_max = 0, 2 * np.pi/4
        elif np.abs(m1_val - 1e5)<1e-6:
            theta_min, theta_max = 2 * np.pi/4, 2 * np.pi * 2/4
        elif np.abs(m1_val - 1e6)<1e-6:
            theta_min, theta_max = 2 * np.pi * 2/4, 2 * np.pi * 3/4
        elif np.abs(m1_val - 1e7)<1e-6:
            theta_min, theta_max = 2 * np.pi * 3/4, 2 * np.pi * 4/4
        theta = np.linspace(theta_min, theta_max, 100)
        plt.plot(theta, z_orig[i]*np.ones_like(theta),  color=colors[idx], markersize=5, alpha=0.7, linestyle='-', linewidth=2)

# Add scatter points for observations
obs_data = {
    'QPE': (masses_qpe, z_qpe, 'purple', 'D', 6, 0.5),
    'AGN': (smbh_masses, smbh_redshifts, 'k', 'X', 8, 0.5),
    'SDSS': (massbh_sdss, redshift_sdss, 'blue', '.', 8, 0.1)
}

for obs_type, (masses, zs, color, marker, ms, alpha) in obs_data.items():
    theta_list = []
    r_list = []
    for mass, z in zip(masses, zs):
        if mass <= 5e4:
            theta_min, theta_max = 0, np.pi/2
        elif mass <= 5e5:
            theta_min, theta_max = np.pi/2, np.pi
        elif mass <= 5e6:
            theta_min, theta_max = np.pi, 3*np.pi/2
        else:
            theta_min, theta_max = 3*np.pi/2, 2*np.pi
        theta = np.random.uniform(theta_min, theta_max)
        theta_list.append(theta)
        r_list.append(z)
    ax.plot(theta_list, r_list, marker=marker, color=color, alpha=alpha, markersize=ms, linestyle='None', label=obs_type)

# Legend for observations
legend_elements_obs = [
    Line2D([0], [0], marker='D', label='QPE and QPO', alpha=0.5, markerfacecolor='purple', markersize=6, linestyle='None', color='purple'),
    Line2D([0], [0], marker='X', label='AGN', alpha=0.5, markerfacecolor='k', markersize=8, linestyle='None', color='k'),
    Line2D([0], [0], marker='.', label='SDSS Quasars', alpha=0.1, markerfacecolor='blue', markersize=8, linestyle='None', color='blue'),
]
leg_obs = ax.legend(handles=legend_elements_obs, frameon=True, bbox_to_anchor=(0.5, 0.01), loc='center', ncol=3)
ax.add_artist(leg_obs)

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
ax.set_thetagrids([45, 135, 225, 315], labels=[r'$5\times10^4M_\odot$', r'$10^5M_\odot$', r'$10^6M_\odot$', r'$10^7M_\odot$'])

# Legend
legend_elements_m2 = [Line2D([0], [0], marker=None, label=f'{m2:.0f}', linestyle='-', linewidth=2, color=colors[idx]) 
                      for idx, m2 in enumerate(sorted(z_data.keys())) if (m2_filter == 'all' or m2 == m2_filter)]
ax.legend(handles=legend_elements_m2, bbox_to_anchor=(0.5, 1.05), loc='lower center', ncols=4, title=r'Secondary mass $m_2 [M_\odot]$')

plt.tight_layout()
output_filename = f'redshift_horizon_polar_a_{spin_a}_tpl_{tpl_val}.png'
plt.savefig(os.path.join(script_dir, output_filename), dpi=400)
print(f"Polar plot saved: figures/{output_filename}")