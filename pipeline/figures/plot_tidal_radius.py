"""
Cells 26 & 27: Plot tidal radius vs black hole mass for different compact objects.
Two versions: one with tidal radius normalized by BH mass, one in solar radii.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Add parent directory to path and change to pipeline directory for data files
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(script_dir)

# Use the physrev style if available
try:
    plt.style.use(os.path.join(pipeline_dir, 'physrev.mplstyle'))
except:
    pass

from few.utils.geodesic import get_separatrix
from few.utils.constants import MRSUN_SI

# Solar radius in meters
R_sun_m = 6.957e8

# Define dictionary of objects with mass (Msun) and radius (m)
objects = {
    "Earth": {
        "name": "Earth",
        "M_Msun": 3e-6,
        "R_m": 7e-3 * R_sun_m,
        "color": 'purple'
    },
    "hot_jupiter": {
        "name": "Hot Jupiter",
        "M_Msun": 0.00095,
        "R_m": 0.13 * R_sun_m,  # ~1 R_J ≈ 0.1 R_sun
        "color": 'orange'
    },
    "brown_dwarf": {
        "name": "Brown Dwarf",
        "M_Msun": 0.05,
        "R_m": 0.10 * R_sun_m,  # ~1 R_J ≈ 0.1 R_sun
        "color": 'brown'
    },
    "white_dwarf": {
        "name": "White Dwarf",
        "M_Msun": 0.6,
        "R_m": 0.012 * R_sun_m,
        "color": 'grey'
    },
    "neutron_star": {
        "name": "Neutron Star",
        "M_Msun": 1.4,
        "R_m": 12_000,  # 12 km
        "color": 'b'
    }
}

# BH mass range
M_bh = np.logspace(3, 7, 300)

# ============================================
# Figure 1: Tidal radius normalized by BH mass
# ============================================
plt.figure(figsize=(3.25*2, 2))

for object_name in objects:
    R_ = objects[object_name]["R_m"] / MRSUN_SI  # Convert to geometric units
    M_ = objects[object_name]["M_Msun"]
    # Tidal radius (in M units)
    r_t = R_ * (M_bh / M_)**(1/3)
    plt.loglog(M_bh, r_t / M_bh, color=objects[object_name]["color"], 
               label=f"{objects[object_name]['name']} M={M_} $M_\\odot$")

# ISCO radius for non-spinning BH (r = 6M)
r_s = 6 * M_bh
plt.loglog(M_bh, r_s / M_bh, label="Last Stable Orbit Schwarzschild", color='k')

# Fill tidally safe region
plt.fill_between(M_bh, 0.0 * get_separatrix(0.998, 0.0, 1.0), get_separatrix(0.0, 0.0, -1.0), 
                 alpha=0.3, color='green', label="Tidally Safe")

# Sgr A* mass
plt.axvline(4e6, color='r', linestyle=':', label='Sgr $A^*$')

plt.xlabel("Black hole mass $ M \\,[M_\\odot]$")
plt.ylabel("Tidal Radius $/M$")
plt.legend(ncol=1, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "tidal_radius_normalized.png"), dpi=300, bbox_inches='tight')
# plt.show()

print("Figure 1 saved to figures/tidal_radius_normalized.png")

# ============================================
# Figure 2: Tidal radius in solar radii
# ============================================
plt.figure(figsize=(3.25*2, 2))

for object_name in objects:
    R_ = objects[object_name]["R_m"] / R_sun_m  # Convert to solar radii
    M_ = objects[object_name]["M_Msun"]
    # Tidal radius (in solar radii)
    r_t = R_ * (M_bh / M_)**(1/3)
    plt.loglog(M_bh, r_t, color=objects[object_name]["color"], 
               label=f"{objects[object_name]['name']} M={M_} $M_\\odot$")

# ISCO radius for non-spinning BH (in solar radii)
r_s = 6 * M_bh
plt.loglog(M_bh, r_s * MRSUN_SI / R_sun_m, label="Last Stable Orbit Schwarzschild", color='k')

# Fill tidally safe region (in solar radii)
plt.fill_between(M_bh, 
                 0.0 * get_separatrix(0.998, 0.0, 1.0) * M_bh * MRSUN_SI / R_sun_m, 
                 get_separatrix(0.0, 0.0, -1.0) * M_bh * MRSUN_SI / R_sun_m, 
                 alpha=0.3, color='green', label="Tidally Safe")

# Sgr A* mass
plt.axvline(4e6, color='r', linestyle=':', label='Sgr $A^*$')

plt.xlabel("Black hole mass $[M_\\odot]$")
plt.ylabel("Tidal Radius $[R_\\odot]$")
plt.legend(ncol=1, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.ylim(100 * MRSUN_SI / R_sun_m, 5e9 * MRSUN_SI / R_sun_m)
plt.savefig(os.path.join(script_dir, "tidal_radius_solar.png"), dpi=300, bbox_inches='tight')
# plt.show()

print("Figure 2 saved to figures/tidal_radius_solar.png")
