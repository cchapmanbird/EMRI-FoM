import numpy as np
sources = {
    "7400": {
        "m1": 4.4e5,
        "m2": 7.8,
        "epsilon": 1.78e-5,
        "a": 0.988,
        "e0": 0.015,
        "p0": 13.6,
        "redshift": 0.2198,
        "SNR_Kerr": 41,
        "Plunging": "Yes",
        "Prograde": "No",
        "T_plunge_yr": 0.94,
        "eccentricity_class": "Near Circular"
    },
    "Icy Source": {
        "m1": 2.07e6,
        "m2": 11.38,
        "epsilon": 5.5e-6,
        "a": 0.972,
        "e0": 0.140,
        "p0": 4.86,
        "redshift": 0.3528,
        "SNR_Kerr": 61,
        "Plunging": "Yes",
        "Prograde": "Yes",
        "T_plunge_yr": 1.42,
        "eccentricity_class": "Near Circular"
    },
    "Chilled source": {
        "m1": 2e6,
        "m2": 45.2,
        "epsilon": 2.3e-5,
        "a": 0.95,
        "e0": 0.316,
        "p0": 7.389,
        "redshift": 0.7714,
        "SNR_Kerr": 100,
        "Plunging": "Yes",
        "Prograde": "Yes",
        "T_plunge_yr": 1.95,
        "eccentricity_class": "Low Eccentricity"
    },
    "12779": {
        "m1": 1.05e6,
        "m2": 85,
        "epsilon": 8.1e-5,
        "a": 0.26,
        "e0": 0.3,
        "p0": 12.333,
        "redshift": 0.5019,
        "SNR_Kerr": 45,
        "Plunging": "Yes",
        "Prograde": "No",
        "T_plunge_yr": 0.7,
        "eccentricity_class": "Low Eccentricity"
    },
    "758": {
        "m1": 3.0e5,
        "m2": 7.9,
        "epsilon": 2.6e-5,
        "a": 0.982,
        "e0": 0.49,
        "p0": 14.5,
        "redshift": 0.2792,
        "SNR_Kerr": 51,
        "Plunging": "No",
        "Prograde": "Yes",
        "T_plunge_yr": 2.3,
        "eccentricity_class": "Moderate Eccentricity"
    },
    "Funky source": {
        "m1": 5.8e6,
        "m2": 5.8,
        "epsilon": 1e-6,
        "a": 0.998,
        "e0": 0.425,
        "p0": 2.12,
        "redshift": 0.7007,
        "SNR_Kerr": 30.5,
        "Plunging": "No",
        "Prograde": "Yes",
        "T_plunge_yr": ">2",
        "eccentricity_class": "Moderate Eccentricity"
    },
    "2085": {
        "m1": 3.8e5,
        "m2": 56,
        "epsilon": 1.5e-4,
        "a": 0.5,
        "e0": 0.74,
        "p0": 15.7,
        "redshift": 0.7582,
        "SNR_Kerr": 120,
        "Plunging": "Yes",
        "Prograde": "Yes",
        "T_plunge_yr": 1.17,
        "eccentricity_class": "Highly Eccentric"
    },
    "12562": {
        "m1": 2.5e5,
        "m2": 22.6,
        "epsilon": 9.2e-5,
        "a": 0.969,
        "e0": 0.77,
        "p0": 16.8,
        "redshift": 0.6208,
        "SNR_Kerr": 67,
        "Plunging": "Yes",
        "Prograde": "Yes",
        "T_plunge_yr": 1.8,
        "eccentricity_class": "Highly Eccentric"
    }
}
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux

traj = EMRIInspiral(func=KerrEccEqFlux)
sources_intr = []

for source, params in sources.items():
    m1 = params["m1"]
    m2 = params["m2"]
    a = params["a"]
    e0 = params["e0"]
    p0 = params["p0"]
    redshift = params["redshift"]
    SNR_Kerr = params["SNR_Kerr"]
    Plunging = params["Plunging"]
    Prograde = params["Prograde"]
    T_plunge_yr = params["T_plunge_yr"]
    eccentricity_class = params["eccentricity_class"]

    M = m1
    mu = m2
    x0 = 1.0
    There = T_plunge_yr if isinstance(T_plunge_yr, float) else 2.0
    t, p, e, x, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, x0, dt=10., T=There, integrate_backwards=False)

    print("--------------------------------------")
    print(f"Source: {source}")
    print(f"t final: {t[-1]}")
    print(f"p final: {p[-1]}")
    print(f"e final: {e[-1]}")
    print(f"x final: {x[-1]}")
    print(f"Phi_phi final: {Phi_phi[-1]}")
    print(f"Phi_r final: {Phi_r[-1]}")
    print(f"Phi_theta final: {Phi_theta[-1]}")

    psd_file = "psd_file_placeholder"
    model = "model_placeholder"
    channels = "channels_placeholder"
    dt = 10.0
    Nmonte = 3
    dev = "cpu"
    thr_snr = 10.0
    thr_err = 0.1

    sources_intr.append({
        "M": M,
        "mu": mu,
        "a": a,
        "e_f": e[-1],
        "T": There,
        "redshift": redshift,
        "repo": source,
    })