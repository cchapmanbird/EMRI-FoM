import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import root_scalar

# Jon https://arxiv.org/pdf/gr-qc/0405137
def get_F_value(mtwo):
    """
    Get the F parameter based on secondary mass (mtwo).
    
    Parameters:
    -----------
    mtwo : float
        Secondary mass in solar masses
        
    Returns:
    --------
    F : float
        F parameter for the given mass type
    """
    if mtwo <= 1.0:
        # white dwarfs
        return 0.2
    elif mtwo <= 50.0:
        # black holes of ~10 Msun
        return 0.03
    else:
        # black holes of ~100 Msun
        return 4e-5

plt.figure()
mtwo = 10**np.linspace(-1, 4)
f_values = [get_F_value(m_) for m_ in mtwo]
plt.loglog(mtwo, f_values, label='F parameter from Gair et al. 2004')
plt.loglog(mtwo, mtwo * 0.1 * (mtwo/1)**(-2), '--', label=r'$0.1 (m_2/1 M_\odot)^{-2}$')
plt.xlabel("Secondary Mass $m_2$ [$M_\\odot$]")
plt.ylabel("F parameter")
plt.legend()
plt.show()

def get_merger_rate(mone, mtwo, M_star = 3e6):
    # F = get_F_value(mtwo)
    F = 0.1 * (mtwo/1)**(-2)
    capture_rate = 1e-4 * F * (mone/M_star)**(3/8) * (mtwo)**(-0.5) # per year
    space_density = 1.7e-3 # Mpc^-3 -> 1Mpc = 1e-3 Gpc -> Mpc^-3 = 1e9 Gpc^-3
    MergerRate = capture_rate * space_density * 1e9
    return MergerRate # Gpc^-3 yr^-1

mone = 1e6
mtwo = 1.0
MergerRate = get_merger_rate(mone, mtwo)
print("Merger Rate", MergerRate ,"Gpc^-3 yr^-1") # in agreement with Table 2 of the paper https://arxiv.org/pdf/gr-qc/0405137
fiducial_distance = 1.0
snr_thr = 30.0
snr_test = 20
N_det = MergerRate * 4 * np.pi/3 * fiducial_distance**3 * (snr_test/snr_thr)
print("Detections in one year", N_det)
# Create grid of masses for plotting
m1_range = np.logspace(4.5, 7, 50)  # 10^4 to 10^8 solar masses
m2_range = np.logspace(-1.0, 4, 50)  # 0.1 to 10^4 solar masses
M1_grid, M2_grid = np.meshgrid(m1_range, m2_range)

# Calculate merger rates for the grid
merger_rate_grid = np.zeros_like(M1_grid)
for i in range(len(m1_range)):
    for j in range(len(m2_range)):
        merger_rate_grid[j, i] = get_merger_rate(M1_grid[j, i], M2_grid[j, i])

plt.figure(figsize=(8, 6))
contour = plt.contourf(M1_grid, M2_grid, np.log10(4 * np.pi/3 * fiducial_distance**3 * merger_rate_grid), levels=20, cmap='viridis')
plt.colorbar(contour, label='Log Merger Rate [Gpc$^{-3}$ yr$^{-1}$]')
plt.xlabel('Primary Mass $m_1$ [M$_\odot$]')
plt.ylabel('Secondary Mass $m_2$ [M$_\odot$]')
plt.xscale('log')
plt.yscale('log')
plt.title('EMRI Merger Rate')
plt.tight_layout()
plt.show()

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import root_scalar

# Observation parameters
Tobs = 1.0        # years
SNR_det = 30.0

# Horizon distance function
def horizon_distance(snr, snr_ref=20.0, d_ref=5.0):
    return d_ref * snr_ref / snr  # Gpc

# Maximum redshift corresponding to horizon
def z_horizon(snr):
    d_h = horizon_distance(snr) * 1e3  # Gpc -> Mpc
    sol = root_scalar(lambda z: cosmo.luminosity_distance(z).value - d_h,
                      bracket=[0, 10], method='bisect')
    return sol.root

# Differential number of detections per redshift
def dNdz(z):
    dVc_dz = cosmo.differential_comoving_volume(z).value * 4*np.pi / 1e9  # Gpc^3
    return MergerRate * dVc_dz / (1 + z)

# Integrate
z_max = z_horizon(SNR_det)
N_det, _ = quad(dNdz, 0, z_max)
N_det *= Tobs

print(f"Estimated number of detectable sources: {N_det:.2f}")