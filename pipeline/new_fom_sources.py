import numpy as np
import matplotlib.pyplot as plt
import itertools
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.constants import MTSUN_SI, YRSID_SI
from few.utils.utility import get_p_at_t
import itertools
from few.utils.geodesic import get_separatrix
from scipy.optimize import root_scalar
import json
traj = EMRIInspiral(func=KerrEccEqFlux)

def find_semi_latus_rectum(M, mu, a, e_2yr, T=2.0):
    sign = np.sign(a)
    if sign == 0.0:
        sign = 1.0
    p_sep = get_separatrix(np.abs(a), e_2yr, sign * 1.0)+0.1
    p_2yr = get_p_at_t(
            traj,
            T,
            [M,mu,np.abs(a),e_2yr,sign * 1.0,],
            index_of_a=2,
            index_of_p=3,
            index_of_e=4,
            index_of_x=5,
            traj_kwargs={},
            xtol=2e-6,
            rtol=8.881784197001252e-6,
            bounds=[p_sep, 200.0],
        )
    return p_2yr
    # except:
    #     return None

# Generate a grid of sources
m1_values = [1e7, 1e6, 1e5, 1e4]
m2_values = [1e1, 1e2, 1e3]
# mass_ratio = [1e-6, 1e-5, 1e-4, 1e-3]
a_values = [-0.9, 0.0, 0.9]
e_2yr_values = [1e-4, 0.1, 0.2, 0.5] # Eccentricity values
grid_sources = {}
T_plunge_values = [1.99]#[7/365, 30/365, 1.0, 2.0] # List of time to plunge values
z = 1.0
i = 0
produce_sources = False
if produce_sources:
    plt.figure(figsize=(6, 4))
    for m1, m2, a, e in itertools.product(m1_values, m2_values, a_values, e_2yr_values):
        m1 = m1 * (1 + z)
        m2 = m2 * (1 + z)
        if m2 < 1.0:
            break
        
        p0 = None
        for T_plunge in T_plunge_values:
            print(f"m1: {m1}, m2: {m2}, a: {a}, e: {e}, T_plunge: {T_plunge}")
            try:
                p0 = find_semi_latus_rectum(m1, m2, a, e, T=T_plunge)
                print(f"T_last: {T_last}, T_plunge: {T_plunge}")
            except:
                print(f"Error integrating forwards {m1}, {m2}, {a}, {e}, T_plunge: {T_plunge}. Skipping...")
            

        if p0 is not None:
            # run forward
            sign = np.sign(a)
            if sign == 0.0:
                sign = 1.0
            t, p_temp, e_temp, x_temp, Phi_phi, Phi_r, Phi_theta = traj(m1, m2, np.abs(a), p0, e, sign*1.0, dt=1e-4, T=10., integrate_backwards=False)
            T_last = t[-1] / YRSID_SI
            e_f = e_temp[-1]
            p_f = p_temp[-1]
            omegaPhi, omegaTheta, omegaR = get_fundamental_frequencies(np.abs(a), p_temp, e_temp, x_temp)
            dimension_factor = 2.0 * np.pi * m1 * MTSUN_SI
            omegaPhi = omegaPhi / dimension_factor
            omegaTheta = omegaTheta / dimension_factor
            omegaR = omegaR / dimension_factor
            initial_frequency = 2 * omegaPhi[0]
            final_frequency = 2 * omegaPhi[-1]
            plt.loglog(np.abs(2 * omegaPhi), e_temp, '-.', label=f"Source_{i+1}", alpha=0.7)
            # plt.plot(p_temp, e_temp, label=f"m1: {m1}, m2: {m2}, a: {a}, e_f: {e}")
            
            # print(f"m1: {m1}, m2: {m2}, a: {a}, e_f: {e}")
            print(f"Final T_last: {T_last}")
            source_name = f"Source_{i+1}"
            grid_sources[source_name] = {
                "m1": m1,
                "m2": m2,
                "a": a,
                "e0": e,
                "p0": p0,
                "e_f": e_f,
                "p_f": p_f,
                "initial_frequency": initial_frequency,
                "final_frequency": final_frequency,
                "redshift": z,  # Default redshift
                "T_plunge_yr": T_last,  # Time to plunge
            }
            i += 1
    plt.xlabel("GW frequency [Hz]")
    plt.ylabel("eccentricity")
    plt.tight_layout()
    plt.savefig(f"trajectory.png", dpi=300)
    # breakpoint()
    # Save grid_sources to a JSON file
    with open(f"fom_sources_e2yr.json", "w") as json_file:
        json.dump(grid_sources, json_file, indent=4)

# open the JSON file and read the data
with open(f"fom_sources_e2yr.json", "r") as json_file:
    grid_sources = json.load(json_file)



# nuPlunge = catalog[:, 4]
# gamPlunge = catalog[:, 5]
# phiPlunge = catalog[:, 6]
# costhetaSky = catalog[:, 7]
# phiSky = catalog[:, 8]
# cosLambda = catalog[:, 9]
# alpPlunge = catalog[:, 10]
# SMBHspin = catalog[:, 11]
# costhetaSpin = catalog[:, 12]
# phiSpin = catalog[:, 13]
# Zeta = catalog[:, 14]
# SMBHQuad = catalog[:, 15]
# SNRI = catalog[:, 16]
# SNRII = catalog[:, 17]
# SNR_tot = catalog[:, 18]

# breakpoint()
# plot sources Mass1 vs Mass2
plt.figure(figsize=(4, 4))
for cat_name in ["/M11/EMRICAT101_MBH10_SIGMA2_NPL1100_CUSP2_JON1_SPIN3_EuNK0_SNR.dat"]:#"M1/EMRICAT101_MBH10_SIGMA2_NPL1010_CUSP1_JON2_SPIN1_EuNK0_SNR.dat",
    catalog = np.loadtxt("/data/lsperi/EMRI-FoM/lisa-fom/catalogs/EMRIs/"+cat_name,skiprows=1)
    # M11/
    tPlunge = catalog[:, 0]
    logmu = catalog[:, 1]
    logM = catalog[:, 2]
    ePlunge = catalog[:, 3]

    Mcat = np.exp(logM)/MTSUN_SI
    mucat = np.exp(logmu)/MTSUN_SI

    plt.scatter(Mcat, mucat, color='blue', alpha=0.1)

for source, params in grid_sources.items():
    m1 = params["m1"]/(1+z)
    m2 = params["m2"]/(1+z)
    plt.scatter(m1, m2, alpha=0.7, color='red', marker='X')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass 1 (Solar Masses)")
plt.ylabel("Mass 2 (Solar Masses)")
plt.grid()
plt.tight_layout()
plt.savefig("mass1_vs_mass2.png", dpi=300)

# Create a plot for the eccentricities
plt.figure(figsize=(4, 4))
mass_ratios = [params["m2"] / params["m1"] for params in grid_sources.values()]
final_eccentricities = [params["e_f"] for params in grid_sources.values()]
plt.scatter(mass_ratios, final_eccentricities, alpha=0.7, color='red', marker='X')

for cat_name in ["M1/EMRICAT101_MBH10_SIGMA2_NPL1010_CUSP1_JON2_SPIN1_EuNK0_SNR.dat","M11/EMRICAT101_MBH10_SIGMA2_NPL1100_CUSP2_JON1_SPIN3_EuNK0_SNR.dat"]:
    catalog = np.loadtxt("/data/lsperi/EMRI-FoM/lisa-fom/catalogs/EMRIs/"+cat_name,skiprows=1)
    # M11/
    tPlunge = catalog[:, 0]
    logmu = catalog[:, 1]
    logM = catalog[:, 2]
    ePlunge = catalog[:, 3]
    Mcat = np.exp(logM)/MTSUN_SI
    mucat = np.exp(logmu)/MTSUN_SI

    plt.scatter(mucat/Mcat, ePlunge, color='blue', alpha=0.1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass Ratio (m2/m1)")
plt.ylabel("Final Eccentricity")
plt.title("Mass Ratio vs Final Eccentricity")
plt.grid(True)
plt.tight_layout()
plt.savefig("mass_ratio_vs_final_eccentricity.png", dpi=300)

# plot for spin and eccentricities

plt.figure(figsize=(4, 4))
spins = np.asarray([params["a"] for params in grid_sources.values()])
final_eccentricities = np.asarray([params["e_f"] for params in grid_sources.values()])
# select only the sources with a > 0.0
mask_positive = spins >= -1
plt.scatter(spins[mask_positive], final_eccentricities[mask_positive], alpha=0.7, color='red', marker='P')

for cat_name in ["M1/EMRICAT101_MBH10_SIGMA2_NPL1010_CUSP1_JON2_SPIN1_EuNK0_SNR.dat", "M11/EMRICAT101_MBH10_SIGMA2_NPL1100_CUSP2_JON1_SPIN3_EuNK0_SNR.dat"]:
    catalog = np.loadtxt("/data/lsperi/EMRI-FoM/lisa-fom/catalogs/EMRIs/" + cat_name, skiprows=1)
    tPlunge = catalog[:, 0]
    logmu = catalog[:, 1]
    logM = catalog[:, 2]
    ePlunge = catalog[:, 3]
    Mcat = np.exp(logM) / MTSUN_SI
    mucat = np.exp(logmu) / MTSUN_SI
    SMBHspin = catalog[:, 11]

    plt.scatter(SMBHspin, ePlunge, color='blue', alpha=0.1)

plt.xlabel("Spin (a)")
plt.ylabel("Final Eccentricity")
plt.title("Final Eccentricity vs Spin")
# plt.xscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig("final_eccentricity_vs_spin.png", dpi=300)

# plt.figure()
# for source, params in grid_sources.items():
    
#     m1 = params["m1"]
#     m2 = params["m2"]
#     a = params["a"]
#     e_f = params["e_f"]
#     redshift = params["redshift"]
#     T_plunge_yr = params["T_plunge_yr"]
#     print("--------------------------------------")
#     print(f"Source: {source}")
#     print(f"m1: {m1}")
#     print(f"m2: {m2}")
#     print(f"mass ratio: {m2/m1}")
#     print(f"a: {a}")
#     print(f"e_f: {e_f}")
#     M = m1
#     mu = m2
#     x0_f = 1.0
#     p_f = get_separatrix(a, e_f, x0_f) + 0.1
    
#     There = T_plunge_yr if isinstance(T_plunge_yr, float) else 2.0
#     try:
#         t_back, p_back, e_back, x_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(M, mu, a, p_f, e_f, x0_f, dt=1e-4, T=There, integrate_backwards=True)
#     except:
#         print(f"Error integrating backwards for source {source}. Skipping...")
#         continue
#     p0 = p_back[-1]
#     e0 = e_back[-1]
#     print(f"p0: {p0}")
#     print(f"e0: {e0}")
#     t, p, e, x, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, x_back[-1], dt=10., T=There, integrate_backwards=False)
#     omegaPhi, omegaTheta, omegaR = get_fundamental_frequencies(a, p, e, x)
    
#     dimension_factor = 2.0 * np.pi * M * MTSUN_SI
#     omegaPhi = omegaPhi / dimension_factor
#     omegaTheta = omegaTheta / dimension_factor
#     omegaR = omegaR / dimension_factor
#     initial_frequency = 2 * omegaPhi[0]
#     final_frequency = 2 * omegaPhi[-1]
#     plt.loglog(np.abs(2 * omegaPhi), e, '-.', label=source, alpha=0.7)


#     psd_file = "psd_file_placeholder"
#     model = "model_placeholder"
#     channels = "channels_placeholder"
#     dt = 10.0
#     Nmonte = 3
#     dev = "cpu"
#     thr_snr = 10.0
#     thr_err = 0.1

#     sources_intr.append({
#         "m1_source": m1,
#         "m2_source": m2,
#         "m1_detector": m1 * (1 + redshift),
#         "m2_detector": m2 * (1 + redshift),
#         "a": a,
#         "e_final": e[-1],
#         "p_final": p[-1],
#         "p_initial": p0,
#         "e_initial": e0,
#         "T_inspiral": There,
#         "redshift": redshift,
#         "repo": source,
#         "initial_frequency": initial_frequency,
#         "final_frequency": final_frequency,
#     })
# plt.xlabel("p")
# plt.xlabel("GW frequency (Hz)")
# plt.ylabel("e")
# # plt.legend()
# plt.grid()
# plt.tight_layout()
# # plt.savefig("p_vs_e.png")
# plt.savefig("frequency_vs_e.png")
# breakpoint()
# sources_intr = grid_sources.values()
# # Extract data for plotting
# mu_values = [src["mu"] for src in sources_intr]
# M_values = [src["M"] for src in sources_intr]
# initial_frequencies = [src["initial_frequency"] for src in sources_intr]
# final_frequencies = [src["final_frequency"] for src in sources_intr]
# e_f_values = [src["e_f"] for src in sources_intr]
# e_i_values = [src["e0"] for src in sources_intr]
# redshift_values = [src["redshift"] for src in sources_intr]

# # Plot 1: mu vs M
# plt.figure(figsize=(8, 6))
# plt.scatter(M_values, mu_values, color='blue', alpha=0.7)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("M (Solar Masses)")
# plt.ylabel("mu (Solar Masses)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("mu_vs_M.png")
# plt.close()

# # Plot 3: M vs Redshift
# plt.figure(figsize=(8, 6))
# plt.scatter(M_values, redshift_values, color='purple', alpha=0.7)
# plt.xscale("log")
# plt.ylabel("Redshift")
# plt.xlabel("M (Solar Masses)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("M_vs_redshift.png")
# plt.close()