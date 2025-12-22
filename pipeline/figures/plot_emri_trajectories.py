"""
Plot EMRI trajectories in time-frequency and frequency-eccentricity space.
Combines Cell 23 and Cell 24 from the analysis notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import os
import sys
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.utils.geodesic import get_fundamental_frequencies, get_separatrix
from few.utils.constants import MTSUN_SI, YRSID_SI, GM_SUN, MRSUN_SI
from few.utils.utility import get_p_at_t

# Add parent directory to path and change to pipeline directory for data files
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(script_dir)
sys.path.insert(0, pipeline_dir)
os.chdir(pipeline_dir)

# Use the physrev style if available
try:
    plt.style.use(os.path.join(pipeline_dir, 'physrev.mplstyle'))
except:
    pass

# -----------------------------------------------------------------------------
# Setup trajectory objects
# -----------------------------------------------------------------------------
# Standard trajectory for reference
traj_std = EMRIInspiral(func="SchwarzEccFlux")

class KerrEccEqFluxPowerLaw(KerrEccEqFlux):
    def modify_rhs(self, ydot, y):
        # in-place modification of the derivatives
        LdotAcc = (
            self.additional_args[0]
            * pow(y[0] / 10.0, self.additional_args[1])
            * 32.0
            / 5.0
            * pow(y[0], -7.0 / 2.0)
        )
        dL_dp = (
            -3 * pow(self.a, 3)
            + pow(self.a, 2) * (8 - 3 * y[0]) * np.sqrt(y[0])
            + (-6 + y[0]) * pow(y[0], 2.5)
            + 3 * self.a * y[0] * (-2 + 3 * y[0])
        ) / (2.0 * pow(2 * self.a + (-3 + y[0]) * np.sqrt(y[0]), 1.5) * pow(y[0], 1.75))
        # transform back to pdot from Ldot and add GW contribution
        # print(ydot[0] , LdotAcc / dL_dp, (LdotAcc / dL_dp)/ydot[0])
        ydot[0] = ydot[0] + LdotAcc / dL_dp


A = 0.0
nr = 8.0
KerrEccEqFluxPowerLaw().additional_args = (A, nr)  # A is the power-law coefficient, nr is the power-law exponent
traj = EMRIInspiral(func=KerrEccEqFluxPowerLaw)
# traj.func.add_fixed_parameters(m1 = 1e6, m2 = 10, a = 0.9, additional_args=(A, nr))
print("All modules loaded successfully!")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def find_semi_latus_rectum(M, mu, a, e0, T=2.0):
    """
    Find the semi-latus rectum (p) for given parameters at a specific time to plunge.
    
    Parameters:
    -----------
    M : float
        Primary mass in solar masses
    mu : float
        Secondary mass in solar masses
    a : float
        Dimensionless spin parameter
    e0 : float
        Initial eccentricity
    T : float
        Time to plunge in years
    
    Returns:
    --------
    p_T : float
        Semi-latus rectum value
    """
    sign = np.sign(a)
    if sign == 0.0:
        sign = 1.0
    
    # Get separatrix (minimum stable orbit)
    p_sep = get_separatrix(np.abs(a), e0, sign * 1.0) + 0.1
    
    # Find p at time T before plunge
    p_T = get_p_at_t(
        traj,
        T,
        [M, mu, np.abs(a), e0, sign * 1.0],
        index_of_a=2,
        index_of_p=3,
        index_of_e=4,
        index_of_x=5,
        traj_kwargs={},
        xtol=2e-6,
        rtol=8.881784197001252e-6,
        bounds=[p_sep, 200.0],
    )
    return p_T


def get_traj_out(m1, m2, a, ef, T, z=0.0):
    """
    Compute EMRI trajectory for given parameters.

    Parameters:
    -----------
    m1 : float
        Primary mass in solar masses
    m2 : float
        Secondary mass in solar masses
    a : float
        Dimensionless spin parameter (positive for prograde, negative for retrograde)
    ef : float
        Final eccentricity at plunge
    T : float
        Observation time in years
    z : float
        Redshift (optional, default 0.0)
        
    Returns:
    --------
    t_back : array
        Time array
    p_back : array
        Semi-latus rectum evolution
    e_back : array
        Eccentricity evolution
    x_back : array
        Inclination parameter evolution
    frequency_gw_2phi : array
        GW frequency (2 * orbital frequency)
    """
    x0_f = 1.0 * np.sign(a) if a != 0.0 else 1.0
    a = np.abs(a)
    
    print(f"Computing trajectory: m1={m1}, m2={m2}, a={a * x0_f}, ef={ef}, T={T}")
    
    p_f = traj.func.min_p(ef, x0_f, a)
    
    try:
        # Forward integration to find initial conditions
        t_forward, p_forward, e_forward, x_forward, Phi_phi_forward, Phi_r_forward, Phi_theta_forward = traj(
            m1, m2, a, p_f, ef, x0_f, A, nr, dt=1e-5, T=100.0, integrate_backwards=False
        )
        
        # Backward integration for observation window
        t_back, p_back, e_back, x_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(
            m1, m2, a, p_forward[-1], e_forward[-1], x_forward[-1], A, nr, dt=1e-5, T=T, integrate_backwards=True
        )
        
        print(f"  Final eccentricity: {e_forward[-1]:.6f}, Initial eccentricity: {e_back[0]:.6f}")
        print(f"  Time span: {t_back[-1]/YRSID_SI:.2f} to {t_back[0]/YRSID_SI:.2f} years")
        
        # Compute GW frequencies
        omegaPhi, omegaTheta, omegaR = get_fundamental_frequencies(a, p_back, e_back, x_back)
        dimension_factor = 2.0 * np.pi * m1 * MTSUN_SI
        omegaPhi = omegaPhi / dimension_factor
        frequency_gw_2phi = np.abs(2 * omegaPhi)  # Dominant GW frequency
        
        print(f"  Frequency range: {frequency_gw_2phi[0]:.6e} to {frequency_gw_2phi[-1]:.6e} Hz")
        
        return t_back, p_back, e_back, x_back, frequency_gw_2phi
    except Exception as e:
        print(f"  Error computing trajectory: {e}")
        return None, None, None, None, None


# -----------------------------------------------------------------------------
# Source configurations
# -----------------------------------------------------------------------------
# (m1, m2, dt, T, a, e_f) - primary mass, secondary mass, dt, observation time, spin, final eccentricity
source_configs = []

# Mass ratio 1e-3 sources with circular orbits
for a in [0.99, -0.99]:
    source_configs.append((5e4, 50.0, 0.6, 0.25, a, 0.0))
    source_configs.append((1e5, 100.0, 0.6, 0.25, a, 0.0))
    source_configs.append((1e6, 1000.0, 10.0, 0.25, a, 0.0))
    source_configs.append((1e7, 10000.0, 100.0, 0.25, a, 0.0))

# Eccentric sources for frequency-eccentricity plot
for a in [0.99, -0.99]:
    for ef in [0.0, 0.01]:
        source_configs.append((5e4, 50.0, 0.6, 0.25, a, ef))
        source_configs.append((1e5, 100.0, 0.6, 0.25, a, ef))
        source_configs.append((1e6, 1000.0, 10.0, 0.25, a, ef))
        source_configs.append((1e7, 10000.0, 100.0, 0.25, a, ef))

# -----------------------------------------------------------------------------
# Compute trajectories
# -----------------------------------------------------------------------------
print("=" * 60)
print("Computing EMRI trajectories...")
print("=" * 60)

store_results = {}
for idx, (m1, m2, dt, T, a, ef) in enumerate(source_configs):
    result = get_traj_out(m1, m2, a, ef, T)
    if result[0] is not None:
        store_results[idx] = {
            "m1": m1,
            "m2": m2,
            "a": a,
            "e_f": ef,
            "Tpl": T,
            "dt": dt,
            "results": result
        }
    print()

print(f"Successfully computed {len(store_results)} trajectories")

# -----------------------------------------------------------------------------
# Get unique m1 values and create color mapping
# -----------------------------------------------------------------------------
m1_values_unique = sorted(set([v["m1"] for v in store_results.values()]))

cmap_continuous = cm.get_cmap('cividis')
norm = plt.Normalize(vmin=0, vmax=len(m1_values_unique) - 1)

color_dict = {}
for idx, m1 in enumerate(m1_values_unique):
    color_dict[m1] = cmap_continuous(norm(idx))

colors = [color_dict[m1] for m1 in m1_values_unique]
cmap_discrete = ListedColormap(colors)

# -----------------------------------------------------------------------------
# PLOT 1: Time-Frequency Space
# -----------------------------------------------------------------------------
print("\nGenerating time-frequency plot...")
fig1 = plt.figure(figsize=(3.25, 2*1.5))

sm = cm.ScalarMappable(cmap=cmap_discrete, norm=plt.Normalize(vmin=0, vmax=len(m1_values_unique) - 1))
sm.set_array([])

for key, values in store_results.items():
    t_back, p_back, e_back, _, frequency_gw_2phi = values["results"]
    a = values["a"]
    e_f = values["e_f"]
    m_1 = values["m1"]
    m_2 = values["m2"]
    
    color = color_dict[m_1]
    linestyle = '-' if a > 0 else '--'
    
    # Only plot sources with mass ratio ~1e-3 and zero eccentricity
    mass_ratio = m_2 / m_1
    if np.isclose(mass_ratio, 1e-3, rtol=0.1) and e_f == 0.0:
        plt.plot((t_back[0] - t_back + t_back[-1]) / YRSID_SI, frequency_gw_2phi,
                 color=color, alpha=0.9, linestyle=linestyle, linewidth=2)

plt.xlabel("Time [years]")
plt.ylabel("GW Frequency [Hz]")
plt.grid(True, alpha=0.3)
plt.yscale('log')

cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02, ticks=range(len(m1_values_unique)))
lab = [f'$10^{int(np.log10(m))}$' for m in m1_values_unique]
if m1_values_unique[0] == 5e4:
    lab[0] = r'$5\times 10^{4}$'
cbar.ax.set_yticklabels(lab)
cbar.set_label(r'$m_1$ [$M_\odot$]',labelpad=-6)
plt.ylim(1e-4, 1.0)

# Add Nyquist frequency lines
dt_values = sorted(set([v["dt"] for v in store_results.values()]))
nyquist_f = 1 / (2 * np.array(dt_values))
linestyles = ['-', '-.', ':']
for ff, ls in zip(nyquist_f, linestyles[:len(nyquist_f)]):
    plt.axhline(ff, color='r', linestyle=ls, label=f'$\\Delta t={1/(2*ff):.1f}$ s')

# Legends
nyquist_legend = plt.legend(title='Nyquist Frequencies', bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=3, frameon=False)
plt.gca().add_artist(nyquist_legend)
line_legend_elements = [
    Line2D([0], [0], color='k', linestyle='-', linewidth=2, label='Prograde $a = +0.99$'),
    Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Retrograde $a = -0.99$'),
]
plt.legend(handles=line_legend_elements, bbox_to_anchor=(0.5, 1.18), loc='lower center', ncol=2, frameon=False)

plt.tight_layout()

plt.savefig(os.path.join(script_dir, "emri_trajectories_time_frequency.png"), dpi=300, bbox_inches='tight')
print("Saved: figures/emri_trajectories_time_frequency.png")

# -----------------------------------------------------------------------------
# PLOT 2: Frequency-Eccentricity Space
# -----------------------------------------------------------------------------
print("\nGenerating frequency-eccentricity plot...")

fig2 = plt.figure()

sm2 = cm.ScalarMappable(cmap=cmap_discrete, norm=plt.Normalize(vmin=0, vmax=len(m1_values_unique) - 1))
sm2.set_array([])

for key, values in store_results.items():
    t_back, _, e_back, _, frequency_gw_2phi = values["results"]
    a = values["a"]
    e_f = values["e_f"]
    m_1 = values["m1"]
    m_2 = values["m2"]
    Tpl = values["Tpl"]
    
    mass_ratio = m_2 / m_1
    if not np.isclose(mass_ratio, 1e-3, rtol=0.1):
        continue
    
    color = color_dict.get(m_1, 'gray')
    linestyle = '-' if a > 0 else '--'
    
    if (Tpl == 0.25)and(e_f == 0.01):
        plt.plot(frequency_gw_2phi, e_back, linestyle, color=color, alpha=0.7, linewidth=2)
    
    # # Add markers for starting points
    # if a > 0.0:
    #     plt.scatter(frequency_gw_2phi[0], e_back[0], color=color, marker='^', s=80)
    # else:
    #     plt.scatter(frequency_gw_2phi[0], e_back[0], color=color, marker='v', s=80)

plt.xlabel("GW Frequency [Hz]")
plt.ylabel("Eccentricity")
plt.grid(True, alpha=0.3, which='both')

cbar2 = plt.colorbar(sm2, ax=plt.gca(), pad=0.02, ticks=range(len(m1_values_unique)))
lab2 = [f'$10^{int(np.log10(m))}$' for m in m1_values_unique]
if m1_values_unique[0] == 5e4:
    lab2[0] = r'$5\times 10^{4}$'
cbar2.ax.set_yticklabels(lab2)
cbar2.set_label(r'$m_1$ [$M_\odot$]', labelpad=-4)

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-4, 1.0)

legend_elements = [
    Line2D([0], [0], color='k', linestyle='-', linewidth=2, label='Prograde $a = +0.99$'),
    Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Retrograde $a = -0.99$'),
]
plt.legend(handles=legend_elements, ncols=1, bbox_to_anchor=(0.5, 1.05), loc='lower center')

plt.tight_layout()

plt.savefig(os.path.join(script_dir, "emri_trajectories_frequency_eccentricity.png"), dpi=300, bbox_inches='tight')
print("Saved: figures/emri_trajectories_frequency_eccentricity.png")


# -----------------------------------------------------------------------------
# PLOT 1 and 2: Combined Time-Frequency and Frequency-Eccentricity Space
# -----------------------------------------------------------------------------
print("\nGenerating combined time-frequency and frequency-eccentricity plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.25*2, 2*1.1), sharey=True)

sm = cm.ScalarMappable(cmap=cmap_discrete, norm=plt.Normalize(vmin=0, vmax=len(m1_values_unique) - 1))
sm.set_array([])

# Subplot 1: Time-Frequency Space
for key, values in store_results.items():
    t_back, p_back, e_back, _, frequency_gw_2phi = values["results"]
    a = values["a"]
    e_f = values["e_f"]
    m_1 = values["m1"]
    m_2 = values["m2"]
    
    color = color_dict[m_1]
    linestyle = '-' if a > 0 else '--'
    
    # Only plot sources with mass ratio ~1e-3 and zero eccentricity
    mass_ratio = m_2 / m_1
    if np.isclose(mass_ratio, 1e-3, rtol=0.1) and e_f == 0.0:
        ax1.plot((t_back[0] - t_back + t_back[-1]) / YRSID_SI, frequency_gw_2phi,
                 color=color, alpha=0.9, linestyle=linestyle, linewidth=2)

ax1.set_xlabel("Time [years]")
ax1.set_ylabel("GW Frequency [Hz]")
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.set_ylim(1e-4, 1.0)

# Add Nyquist frequency lines to ax1
dt_values = sorted(set([v["dt"] for v in store_results.values()]))
nyquist_f = 1 / (2 * np.array(dt_values))
linestyles = ['-', '-.', ':']
for ff, ls in zip(nyquist_f, linestyles[:len(nyquist_f)]):
    ax1.axhline(ff, color='r', linestyle=ls, label=f'$\\Delta t={1/(2*ff):.1f}$ s')

ax1.legend(title='Nyquist Frequencies', bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=3)

# # Colorbar for ax1
# cbar1 = plt.colorbar(sm, ax=ax1, pad=0.02, ticks=range(len(m1_values_unique)))
# lab = [f'$10^{int(np.log10(m))}$' for m in m1_values_unique]
# if m1_values_unique[0] == 5e4:
#     lab[0] = r'$5\times 10^{4}$'
# cbar1.ax.set_yticklabels(lab)
# cbar1.set_label(r'Primary Mass $M_1$ [$M_\odot$]')

# Subplot 2: Frequency-Eccentricity Space (with y-axis as GW Frequency, x-axis as Eccentricity)
sm2 = cm.ScalarMappable(cmap=cmap_discrete, norm=plt.Normalize(vmin=0, vmax=len(m1_values_unique) - 1))
sm2.set_array([])

for key, values in store_results.items():
    t_back, _, e_back, _, frequency_gw_2phi = values["results"]
    a = values["a"]
    e_f = values["e_f"]
    m_1 = values["m1"]
    m_2 = values["m2"]
    Tpl = values["Tpl"]
    
    mass_ratio = m_2 / m_1
    if not np.isclose(mass_ratio, 1e-3, rtol=0.1):
        continue
    
    color = color_dict.get(m_1, 'gray')
    linestyle = '-' if a > 0 else '--'
    
    if (Tpl == 0.25) and (e_f == 0.01):
        ax2.plot(e_back, frequency_gw_2phi, linestyle, color=color, alpha=0.7, linewidth=2)

ax2.set_xlabel("Eccentricity")
# ax2.set_ylabel("GW Frequency [Hz]")  # Shared y-axis
ax2.grid(True, alpha=0.3, which='both')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlim(4e-3, 1.0)  # Assuming similar range, adjust if needed

legend_elements = [
    Line2D([0], [0], color='k', linestyle='-', linewidth=2, label='Prograde $a = +0.99$'),
    Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Retrograde $a = -0.99$'),
]
ax2.legend(handles=legend_elements, ncols=1, bbox_to_anchor=(0.5, 1.05), loc='lower center')

# Colorbar for ax2 (optional, since shared y, but keeping for consistency)
cbar2 = plt.colorbar(sm2, ax=ax2, pad=0.02, ticks=range(len(m1_values_unique)))
lab2 = [f'$10^{int(np.log10(m))}$' for m in m1_values_unique]
if m1_values_unique[0] == 5e4:
    lab2[0] = r'$5\times 10^{4}$'
cbar2.ax.set_yticklabels(lab2)
cbar2.set_label(r'$m_1$ [$M_\odot$]', labelpad=-4)

plt.tight_layout()

plt.savefig(os.path.join(script_dir, "emri_trajectories_combined.png"), dpi=300, bbox_inches='tight')
print("Saved: figures/emri_trajectories_combined.png")

print("\n" + "=" * 60)
print("Combined plot generated successfully!")
print("=" * 60)

print("\n" + "=" * 60)
print("All plots generated successfully!")
print("=" * 60)
