import os
import logging
import argparse
import numpy as np
from common import standard_cosmology
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import (
    GenerateEMRIWaveform,
)
from few.utils.constants import *

from few.utils.utility import (
    get_separatrix,
)
from few.trajectory.ode import *
from scipy.interpolate import interp1d
from fastlisaresponse import ResponseWrapper
from scipy.interpolate import CubicSpline
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from stableemrifisher.fisher import StableEMRIFisher
import pandas as pd


logger = logging.getLogger()
parser = argparse.ArgumentParser()

# Add command line arguments
parser.add_argument("parameter_file_source", help="Path to a npy file containing source frame parameters at plunge organised as (N_source, N_params_source)")
#N_params_source are organised as (M, mu, a, e_f, i_f, z, T to integrate backwards)
parser.add_argument("parameter_file_detector", help="Path to a npy file to save detector frame parameters at start of observation")

parser.add_argument("--psd_file", help="Path to a file containing PSD frequency-value pairs", default="TDI2_AE_psd.npy")
parser.add_argument("--dt", help="Sampling cadence in seconds", type=float, default=10.0)
parser.add_argument("--use_gpu", help="Whether to use GPU for FIM computation", action="store_true")
parser.add_argument("--N_montecarlo", help="How many random sky localizations to generate", type=int, default=10)
parser.add_argument("--seed", help="numpy seed for random operations.",  action="store_const", const=42)
parser.add_argument("--device", help="GPU device", type=int, default=7)

args = parser.parse_args()

if args.use_gpu:
    import cupy as xp
    print("Using GPU", args.device)
    xp.cuda.Device(args.device).use()
    xp.random.seed(args.seed)
else:
    xp = np


np.random.seed(args.seed)

# Set type of inspiral trajectory
trajectory_class = KerrEccEqFlux

traj = EMRIInspiral(func=trajectory_class)

inspiral_kwargs_back = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e5),  # All of the trajectories will be well under len = 10^5
    "err": 1e-12,  # Set error tolerance on integrator -- RK8
    "integrate_backwards": True,  # Integrate trajectories backwards
}


inspiral_kwargs_forward = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e5),  # all of the trajectories will be well under len = 10^5
    "err": 1e-12,  # Set error tolerance on integrator -- RK8
    "integrate_backwards": False,  # Integrate trajectories forwards
}

# Generate EMRI waveform
base_wave = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", inspiral_kwargs = inspiral_kwargs_forward,  use_gpu=args.use_gpu, sum_kwargs=dict(pad_output=True))

# Order of the langrangian interpolation
order = 25

# Orbit file and kwargs
orbit_file_esa = "../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

# TDI generation
tdi_gen = "2nd generation"
index_lambda = 8
index_beta = 7

# TDI kwargs
tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
)

# PSD 
psd_file = "TDI2_AE_psd.npy"
psdf, psdv = np.load(psd_file).T
psd_interp = CubicSplineInterpolant(psdf, psdv,  use_gpu=args.use_gpu)
psd_wrap = lambda f, **kwargs: psd_interp(f)

source_args_plunge = np.load(args.parameter_file_source)
source_args_initial = np.zeros((len(source_args_plunge), args.N_montecarlo, 17)) #16 = 14 parameters in vacuum + z + Tobs + SNR

z = source_args_plunge[:, 5]

M = source_args_plunge[:, 0] * (1 + z) #detector frame conversion
mu = source_args_plunge[:, 1] * (1 + z) #detector frame conversion
a = source_args_plunge[:, 2]
e_f = source_args_plunge[:, 3]
Y_f = source_args_plunge[:, 4]
p_f = np.array([])
dist = np.array([])
for i in range(len(source_args_plunge)):
    p_f = np.append(p_f, get_separatrix(a[i], e_f[i], Y_f[i])) + 0.1 # add a small number to separatrix to integrate backwards
    dist = np.append(dist, standard_cosmology(H0=67.).dl_zH0(z[i]) / 1000.)

T = source_args_plunge[:, 6]

#Parameters to be varied in the Fisher Matrix for Kerr Eccentric model (problem when e0 = 0)
param_names = [
    'M',
    'mu',
    'a',
    'p0',
    'e0',
    'qS',
    'phiS',
    'qK',
    'phiK',
    'Phi_phi0',
    'Phi_r0',
]

for i in np.arange(len(source_args_plunge)):
    
    # Create a DataFrame for the source parameters
    source_frame_data = {
        "Source": [i],
        "M": [M[i] / (1 + z[i])],
        "μ": [mu[i] / (1 + z[i])],
        "a": [a[i]],
        "p_f": [p_f[i]],
        "e_f": [e_f[i]],
        "i_f": [source_args_plunge[i, 4]],
        "z": [z[i]],
    }

    #Convert the data dictionary into a DataFrame
    df = pd.DataFrame(source_frame_data)

    # Print the DataFrame in a readable format
    print("Source Frame Parameters at Plunge:\n")
    print(df.to_string(index=False, float_format="%.2e"))
    print("\n")

    # Generate random initial phases for the trajectory
    Phi_phi_f = np.random.uniform(0, 2 * np.pi, args.N_montecarlo)
    Phi_theta_f = np.random.uniform(0, 2 * np.pi, args.N_montecarlo)
    Phi_r_f = np.random.uniform(0, 2 * np.pi, args.N_montecarlo)
    
    # Initialize ResponseWrapper model
    model = ResponseWrapper(
        base_wave,
        T[i],
        args.dt,
        index_lambda,
        index_beta,
        t0=100000.,
        flip_hx=True,  # Set to True if waveform is h+ - ihx
        use_gpu=args.use_gpu,
        remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=True,  # Removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )

    # Loop over sky localizations/initial phases
    for j in np.arange(args.N_montecarlo):
        # Integrate trajectory backwards
        t_back, p_back, e_back, Y_back, Phi_phi_back, Phi_r_back, Phi_theta_back = (
        traj(
            M[i],
            mu[i],
            a[i],
            p_f[i],
            e_f[i], 
            Y_f[i],
            Phi_phi0=Phi_phi_f[j],
            Phi_theta0=Phi_theta_f[j],
            Phi_r0=Phi_r_f[j],
            dt=args.dt,
            T=T[i],
            **inspiral_kwargs_back
        )
        )

        p0 = p_back[-1]
        e0 = e_back[-1]
        Y0 = Y_back[-1]
        Phi_phi0 = Phi_phi_back[-1]
        Phi_r0 = Phi_r_back[-1]
        Phi_theta0 = Phi_theta_back[-1]

        # SNR threshold for the Fisher Matrix calculation
        SNR_threshold = 25.0  

        # Define the maximum number of iterations for sky localization generation to avoid infinite loops 
        max_iterations = 100

        iterations = 0
        SNR = 0

        while SNR < SNR_threshold and iterations < max_iterations:
        # Generate random sky localization parameters
            qS = np.pi/2 - np.arcsin(np.random.uniform(-1, 1))
            phiS = np.random.uniform(0, 2 * np.pi)
            qK = np.pi/2 - np.arcsin(np.random.uniform(-1, 1))
            phiK = np.random.uniform(0, 2 * np.pi)

            # Create parameter list
            parameters = [
                M[i], mu[i], a[i], p0, e0, Y0, dist[i], qS, phiS, qK, phiK, 
                Phi_phi0, Phi_theta0, Phi_r0
            ]
    
            # Create StableEMRIFisher object and calculate SNR
            sef = StableEMRIFisher(*parameters, dt=args.dt, T=T[i], EMRI_waveform_gen=model, noise_model=psd_wrap, 
                              noise_kwargs=dict(TDI="TDI2"), param_names=param_names, stats_for_nerds=False, 
                              use_gpu=args.use_gpu, der_order=4)
            SNR = sef.SNRcalc_SEF()  # Calculate the SNR
            print(f"Iteration {iterations + 1}: SNR = {SNR}")
    
            iterations += 1

        if SNR >= SNR_threshold:
            #print(f"Success: SNR = {SNR} exceeded the threshold after {iterations} iterations.")
            source_args_initial[i, j, : -3] = np.array(parameters)
            # Store Tobs and SNR in the last two columns 
            source_args_initial[i, j, -3] = z[i]
            source_args_initial[i, j, -2] = T[i] 
            source_args_initial[i, j, -1] = SNR
        else:
            print(f"Failed to achieve SNR above threshold after {max_iterations} iterations.\n")

# Convert source_args_initial to a DataFrame for better readability in the print
columns = [
    "M", "mu", "a", "p0", "e0", "Y0", "dist", "qS", "phiS", 
    "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0", "z", "T_coalescence", "SNR"
] #Add Tobs and SNR at the end of the columns

# Reshape the 3D array into 2D, keeping track of (Nsource, Nmontecarlo) indices
reshaped_data = source_args_initial.reshape(-1, source_args_initial.shape[-1])

# Generate indices for source type (Nsource) and individual source (Nmontecarlo)
num_sources_per_type = source_args_initial.shape[1]
source_types = np.repeat(np.arange(source_args_initial.shape[0]), num_sources_per_type)
individual_sources = np.tile(np.arange(num_sources_per_type), source_args_initial.shape[0])

# Create the DataFrame and add the indices for clarity
df = pd.DataFrame(reshaped_data, columns=columns)
df.insert(0, "Source Type", source_types)
df.insert(1, "Source Index", individual_sources)

# Compute average SNR for all sources
average_snr = df["SNR"].mean()

# Compute average SNR per source type
average_snr_per_type = df.groupby("Source Type")["SNR"].mean()

# Print the DataFrame
print("\n Detector frame parameters at start of observation:\n")
print(df.to_string(index=False, float_format="%.2e"))


# Print average SNR per source type
print("\nAverage SNR per Source Type:")
print(average_snr_per_type.to_string(float_format="%.2e"))

# Save the data
np.save(args.parameter_file_detector, source_args_initial)