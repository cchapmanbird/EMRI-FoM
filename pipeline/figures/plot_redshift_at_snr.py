#!/usr/bin/env python
"""
Plot: Redshift at given SNR threshold vs primary mass m1.

This script generates a figure showing the redshift reach for different EMRI sources
at a fixed SNR threshold (30), with overlay of electromagnetic observations (QPE, AGN, SDSS Quasars).

Cell 20 from degradation_analysis.ipynb (z_at_snr mode only, SNR threshold = 30)
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
for src_idx in matching_sources:
    m1 = source_metadata[src_idx]['m1']
    m2 = source_metadata[src_idx]['m2']
    
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
# Create plot
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(1, 1, figsize=(3.25, 2*2.0))

colors = plt.cm.tab20(np.linspace(0, 1, len(z_data)))
legend_elements = []

for idx, m2 in enumerate(sorted(z_data.keys())):
    if m2_filter != 'all' and m2 != m2_filter:
        continue
    
    m1_vals = np.array(z_data[m2]['m1'])
    z_orig = np.array(z_data[m2]['z_orig'])
    z_deg = np.array(z_data[m2]['z_deg'])
    
    sort_idx = np.argsort(m1_vals)
    m1_sorted = m1_vals[sort_idx]
    z_orig_sorted = z_orig[sort_idx]
    z_deg_sorted = z_deg[sort_idx]
    
    ax1.plot(m1_sorted, z_orig_sorted, 'o-', color=colors[idx],
             markersize=7, linewidth=1.5, label=f'{m2:.0f}', alpha=0.7)
    if degradation != 1.0:
        ax1.plot(m1_sorted, z_deg_sorted, 's--', color=colors[idx],
                 markersize=6, linewidth=1.5, alpha=0.5)

ax1.set_ylabel(f'Redshift at $\\mathrm{{SNR}}={int(snr_threshold):.0f}$')
ax1.set_yscale('log')

# Overlay EM observations
mask = masses_qpe <= 1e7
ax1.plot(masses_qpe[mask], z_qpe[mask], 'D', color='purple', alpha=0.5, markersize=6, label='QPE and QPO')
ax1.plot(smbh_masses, smbh_redshifts, 'X', color='k', alpha=0.5, markersize=8, label='AGN')
ax1.plot(massbh_sdss, redshift_sdss, '.', color='blue', alpha=0.1, markersize=8, label='SDSS DR16Q Quasars', zorder=0)

legend_elements += [
    Line2D([0], [0], marker='D', label='QPE and QPO', alpha=0.5, markerfacecolor='purple', markersize=6, linestyle='None', color='purple'),
    Line2D([0], [0], marker='X', label='AGN', alpha=0.5, markerfacecolor='k', markersize=8, linestyle='None', color='k'),
    Line2D([0], [0], marker='.', label='SDSS Quasars', alpha=0.1, markerfacecolor='blue', markersize=8, linestyle='None', color='blue'),
]
leg2 = ax1.legend(handles=legend_elements, frameon=True, loc='lower left')
ax1.add_artist(leg2)

ax1.set_ylim(1e-3, 5.)
ax1.set_xlabel(r'Primary mass $m_1 [M_\odot]$')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major')

# Legend for secondary masses
legend_elements_m2 = [Line2D([0], [0], marker='o', label=f'{m2:.0f}', markersize=7, linestyle='-', color=colors[idx]) 
                      for idx, m2 in enumerate(sorted(z_data.keys())) if (m2_filter == 'all' or m2 == m2_filter)]
leg3 = ax1.legend(handles=legend_elements_m2,
                  bbox_to_anchor=(0.95, 1.2),
                  frameon=True, ncols=4,
                  title=r'Secondary mass $m_2 [M_\odot]$')
ax1.set_xlim(4e4, 1.1e7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'z_at_snr.png'), dpi=400)
print("Plot saved: figures/z_at_snr.png")
# plt.show()
