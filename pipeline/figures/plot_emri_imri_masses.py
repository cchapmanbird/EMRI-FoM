"""
Cell 21: Plot of EMRI/IMRI source masses (m1 vs m2) with EM observations context.
Shows FoM sources, QPE/QPO masses, LVK GW observations, and EM BH observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os
import h5py
import sys

# Use the physrev style if available
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    sys.path.insert(0, pipeline_dir)
    plt.style.use(os.path.join(pipeline_dir, 'physrev.mplstyle'))
except:
    pass

from common import CosmoInt

# QPE masses and redshifts from https://arxiv.org/pdf/2404.00941
masses_qpe = np.asarray([1.2, 0.55, 0.55, 3.1, 42.5, 1.8, 5.5, 0.595, 6.55, 88.0, 5.8]) * 1e6
z_qpe = np.asarray([0.0181, 0.0505, 0.0175, 0.024, 0.044, 0.0237, 0.042, 0.13, 0.0206, 0.0136, 0.0053])

# SMBH data from Table EM_measure arXiv-2501.03252v2
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

# Load LVK GW events
with open(os.path.join(pipeline_dir, "lvk_gw_events.json"), "r") as f:
    lvk_gw_events = json.load(f)

# Load stored EMRI sources
with open(os.path.join(pipeline_dir, "so3_sources_Dec8.json"), "r") as f:
    store_results = json.load(f)

# Data from 2020ARA&A..58..257G - Local galaxy measurements
# Green circles (confirmed detections) - distance in Mpc, mass in Msun
local_galaxies = {
    'M32': (0.785, 2.5e6),
    'N5102': (3.4, 9e5),
    'N5206': (3.8, 5.4e5),
    'N4395': (4.3, 3.6e5),
    'N205': (0.824, 2e4),
}

local_galaxy_masses = np.array([v[1] for v in local_galaxies.values()])
local_galaxy_z = np.array([CosmoInt.get_redshift(v[0]/1e3) for v in local_galaxies.values()])

# SDSS DR16Q Quasars
with h5py.File(os.path.join(pipeline_dir, 'sdss_dr16q_quasars.h5'), 'r') as f:
    redshift_sdss = f['redshift'][:]
    log10massbh_sdss = f['log10massbh'][:]
    log10massbh_err_sdss = f['log10massbh_err'][:]
    relative_error_mass = log10massbh_err_sdss * np.log(10)
    mask = (log10massbh_sdss < 7.05) & (relative_error_mass < 0.5)
    redshift_sdss = redshift_sdss[mask]
    log10massbh_sdss = log10massbh_sdss[mask]
    massbh_sdss = 10**log10massbh_sdss

# Compute derived quantities
list_mass = np.asarray([item["mass"] for item in smbh_data]) * 1e6
list_redshift = np.asarray([item["redshift"] for item in smbh_data])
list_name = [item["name"] for item in smbh_data]

min_mass = min(masses_qpe.tolist() + list_mass.tolist())
max_mass = max(masses_qpe.tolist() + list_mass.tolist())

# Create the figure with two subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.25, 2*2), sharex=True, 
                               gridspec_kw={'height_ratios': [1, 1.2], 'hspace': 0.05})

# =============================================================================
# Top subplot: Primary mass vs Redshift
# =============================================================================
# Plot EM observations
# SDSS Quasars (background) - consistent with plot_redshift_at_snr.py
ax1.plot(massbh_sdss, redshift_sdss, '.', color='blue', alpha=0.1, markersize=8, zorder=0, label='SDSS Quasars')

# QPE and QPO data - consistent with plot_redshift_at_snr.py
mask_qpe = masses_qpe <= 1e7
ax1.plot(masses_qpe[mask_qpe], z_qpe[mask_qpe], 'D', color='purple', alpha=0.5, markersize=6, label='QPE and QPO')

# AGN/SMBH data - consistent with plot_redshift_at_snr.py
ax1.plot(list_mass, list_redshift, 'X', color='k', alpha=0.5, markersize=8, label='AGN')

# Local galaxies from 2020ARA&A
ax1.semilogy(local_galaxy_masses, local_galaxy_z, 's', color='orange', alpha=0.5, markersize=6, label='Local Galaxies')

ax1.legend(loc='lower left', fontsize=7)

ax1.set_ylabel("Redshift $z$")
# ax1.set_ylim(1e-3, 5)
ax1.tick_params(axis='x', labelbottom=False)

# =============================================================================
# Bottom subplot: Primary mass vs Secondary mass
# =============================================================================
# Plot filled area for LVK GW observations
ax2.fill_betweenx([min(lvk_gw_events['primary_mass'] + lvk_gw_events['secondary_mass']), 
                   max(lvk_gw_events['primary_mass'] + lvk_gw_events['secondary_mass'])],
                  [3e4], [2e7], color='C1', alpha=0.2, label='GW observations (LIGO-Virgo-KAGRA)')

# Mass ratio lines    
m1_vec = np.logspace(4, 8)
# ax2.loglog(m1_vec, m1_vec * 1, 'r--')
# ax2.text(0.5e5, 2e4, '$m_2/m_1 = 1$', color='r')

ax2.loglog(m1_vec, m1_vec * 1e-3, 'r--')
ax2.text(0.5e5, 4e2, r'$\frac{m_2}{m_1} = 10^{-3}$', color='r')
ax2.loglog(m1_vec, m1_vec * 1e-6, 'r--')
ax2.text(1.5e6, 0.3, r'$\frac{m_2}{m_1} = 10^{-6}$', color='r')

# Plot FoM sources
for key, values in store_results.items():
    m1 = values["m1"]
    m2 = values["m2"]
    Tpl = values["Tpl"]
    if Tpl == 0.25:
        ax2.loglog(m1, m2, 'o', color='k', markersize=4, alpha=0.7)
    # elif Tpl == 1.5:
    #     ax2.loglog(m1, m2, 'ko')
    # elif Tpl == 4.5:
    #     ax2.loglog(m1, m2, 'ko')

# Add label for legend (just one point)
ax2.loglog([], [], 'ko', label='This work EMRIs/IMRIs', markersize=4, alpha=0.7)

ax2.legend(ncols=1, loc='upper left', frameon=False)
ax2.set_xlabel("Primary Mass $m_1$ [$M_\\odot$]")
ax2.set_ylabel("Secondary Mass $m_2$ [$M_\\odot$]")
ax2.set_ylim(1e-1, 18e4)
ax2.set_xlim(3e4, 2e7)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "emri_imri_masses_m1_m2.png"), dpi=300, bbox_inches='tight')
# plt.show()

print(f"min_mass: {min_mass}, max_mass: {max_mass}")
print("Figure saved to figures/emri_imri_masses_m1_m2.png")
