"""
Cell 9: Plot of minimum primary mass vs spin and eccentricity for LISA sensitivity.
Contour plot showing the optimal MBH mass for LISA's most sensitive frequency.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Use the physrev style if available
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    plt.style.use(os.path.join(pipeline_dir, 'physrev.mplstyle'))
except:
    pass

from few.utils.geodesic import get_fundamental_frequencies, get_separatrix
from few.utils.constants import MTSUN_SI

# Minimum sensitive frequency from LISA sensitivity curve
f_sens_min = 7.879418302766868012e-03


def get_M_min_psd(a, e, target_f=7.879418302766868012e-03):
    """
    Get the minimum primary mass that places the GW frequency at LISA's most sensitive frequency.
    
    Parameters:
    -----------
    a : float
        Dimensionless spin parameter (-1 < a < 1)
    e : float
        Eccentricity
    target_f : float
        Target GW frequency (default: LISA's most sensitive frequency)
        
    Returns:
    --------
    M_min : float
        Minimum primary mass in solar masses
    """
    x0 = np.sign(a) * 1.0
    p_sep = get_separatrix(np.abs(a), e, x0) + 1e-6 
    return 2 * np.abs(get_fundamental_frequencies(np.abs(a), p_sep, e, x0)[0]) / (2 * np.pi * target_f * MTSUN_SI)


# Print example values
print("a = 0.99, e = 0.0, m1 = ", get_M_min_psd(0.99, 0.0) / 1e6, " $10^6 M_\\odot$\n")
print("a = -0.99, e = 0.0, m1 = ", get_M_min_psd(-0.99, 0.0) / 1e6, " $10^6 M_\\odot$\n")

# Create a grid for a and e values
a_grid = np.linspace(-0.998, 0.998, 10)
e_grid = np.linspace(0.0001, 0.7, 10)
A_grid, E_grid = np.meshgrid(a_grid, e_grid)

# Calculate minimum mass for each (a, e) combination
M_min_grid = np.zeros_like(A_grid)
for i in range(len(a_grid)):
    for j in range(len(e_grid)):
        M_min_grid[j, i] = get_M_min_psd(A_grid[j, i], E_grid[j, i])

# Convert to log scale
M_min_grid_log = np.log10(M_min_grid)

# Create the contour plot
plt.figure()
contour = plt.contourf(E_grid, A_grid, M_min_grid_log, levels=5, cmap='cividis')
plt.colorbar(contour, label='Log Primary Mass [$\\log_{10}(m_1/M_\\odot)$]')
plt.ylabel('Spin parameter $a$')
plt.xlabel('Eccentricity $e$')

# Add contour lines
cs = plt.contour(E_grid, A_grid, M_min_grid_log, levels=5, colors='k', linewidths=1.0, alpha=0.9)
# Convert log values back to mass values for labels
plt.clabel(cs, inline=True, fontsize=12, fmt=lambda x: rf'$m_1=${10**x/1e5:.0f}$\times 10^5$')
plt.tight_layout()
plt.savefig("best_source_sensitivity.png", dpi=300, bbox_inches='tight')
# plt.show()

print("Figure saved to best_source_sensitivity.pdf and best_source_sensitivity.png")
