import numpy as np
import matplotlib.pyplot as plt
import itertools
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.constants import MTSUN_SI, YRSID_SI
import itertools
from few.utils.geodesic import get_separatrix
from scipy.optimize import root_scalar
import json
traj = EMRIInspiral(func=KerrEccEqFlux)

def get_sep_f(M, a, e_f, x):
    omegaPhi, _, _ = get_fundamental_frequencies(a, get_separatrix(a, e_f, x)+0.1, e_f, x)
    dimension_factor = 2.0 * np.pi * M * MTSUN_SI
    omegaPhi = 2 * omegaPhi / dimension_factor
    return omegaPhi

# Generate a grid of sources
m1_values = [1e7, 1e6, 1e5, 1e4]
m2_values = [10, 100]
# mass_ratio = [1e-6, 1e-5, 1e-4, 1e-3]
a_values = [0.0]
e_f_values = [0.001, 0.01, 0.1, 0.2]
grid_sources = {}
# T_plunge_values = [7/365, 30/365, 1.0, 2.0] # List of time to plunge values
T_plunge_values = [2.0, 1.0, 30/365, 7/365] # List of time to plunge values
T_plunge_values = [1.99]

i = 0
for m1, m2, a, e in itertools.product(m1_values, m2_values, a_values, e_f_values):
    p_f = get_separatrix(a, e, 1.0) + 0.1
    # m2 = ratio * m1
    
    if m2 < 1.0:
        break
    print(f"m1: {m1}, m2: {m2}, a: {a}, e_f: {e}")

    for T_plunge in T_plunge_values:
        T_plunge = T_plunge
        try:
            t_back, p_back, e_back, x_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(m1, m2, a, p_f, e, 1.0, dt=1e-4, T=T_plunge, integrate_backwards=True)
            T_last = T_plunge
            print(f"T_last: {T_last}")
            # run forward
            t, p_temp, e_temp, x, Phi_phi, Phi_r, Phi_theta = traj(m1, m2, a, p_back[-1], e_back[-1], x_back[-1], dt=1e-4, T=10., integrate_backwards=False)
            T_last = t[-1] / YRSID_SI
            plt.figure(figsize=(8, 6))
            plt.plot(p_temp, e_temp, label=f"m1: {m1}, m2: {m2}, a: {a}, e_f: {e}")
            plt.xlabel("p")
            plt.ylabel("e")
            # plt.savefig(f"trajectory_{i+1}.png")
            print(f"T_last: {T_last}, T_plunge: {T_plunge}")
        except:
            # breakpoint()
            print(f"Error integrating backwards {m1}, {m2}, {a}, {e}, T_plunge: {T_plunge}. Skipping...")
            break
        # if T_last == 0.0:
        #     break
    
    print(f"Final T_last: {T_last}")
    source_name = f"Source_{i+1}"
    grid_sources[source_name] = {
        "m1": m1,
        "m2": m2,
        "a": a,
        "e_f": e,
        "redshift": 1.0,  # Default redshift
        "T_plunge_yr": T_last,  # Time to plunge
    }
    i += 1

# Save grid_sources to a JSON file
with open(f"fom_sources_{T_plunge_values[-1]}.json", "w") as json_file:
    json.dump(grid_sources, json_file, indent=4)

# open the JSON file and read the data
with open(f"fom_sources_{T_plunge_values[-1]}.json", "r") as json_file:
    grid_sources = json.load(json_file)
# plot sources Mass1 vs Mass2
plt.figure(figsize=(8, 6))
for source, params in grid_sources.items():
    m1 = params["m1"]
    m2 = params["m2"]
    a = params["a"]
    e_f = params["e_f"]
    redshift = params["redshift"]
    T_plunge_yr = params["T_plunge_yr"]
    plt.scatter(m1, m2, label=source, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass 1 (Solar Masses)")
plt.ylabel("Mass 2 (Solar Masses)")
plt.grid()
plt.savefig("mass1_vs_mass2.png")
plt.figure(figsize=(8, 6))
for source, params in grid_sources.items():
    m1 = params["m1"]
    T_plunge_yr = params["T_plunge_yr"]
    plt.scatter(m1, T_plunge_yr, label=source, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass 1 (Solar Masses)")
plt.ylabel("Time to Plunge (Years)")
plt.grid()
plt.savefig("T_plunge_vs_m1.png")

# sources_intr = []
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