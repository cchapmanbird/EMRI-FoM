import os
import time
import glob
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import io

# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [20.0, 25., 30.]
# qs is used as sky localization threshold
# p0, phi0, theta0 are not used in the threshold

# device: device to use on GPUs
dev = 0
# defines the number of montecarlo runs over phases and sky locations
# N_montecarlo: number of montecarlo runs over phases and sky locations
Nmonte = 10

#define the psd and response properties
channels = 'AET'
tdi2 = True
model = 'scirdv1'
esaorbits = True
psd_file = "TDI2_AE_psd.npy"
# include_foreground: defines whether to include the confusion noise foreground
include_foreground = True

# source frame parameters
# M: central mass of the binary in solar masses
# mu: secondary mass of the binary in solar masses
# a: dimensionless spin of the central black hole
# e_f: final eccentricity of the binary
# T: observation time in years
# z: redshift of the source
# repo: name of the repository where the results will be saved
# psd_file: name of the file with the power spectral density
# dt: time step in seconds
dt = 5.0
sources = []

m1_values = [1e7, 1e6, 1e5, 1e4]
m2 = 10.
a = 0.9
e_2yr_values = [1e-4] # Eccentricity does not have a big impact on horizon
# open dictionary with the sources

for redshift in [0.1, 0.5, 1.0, 2.0]:
    for T_plunge_yr in [0.5, 2.0]:
        for m1 in m1_values:
            for e_f in e_2yr_values:
                source = f"m1={m1}_m2={m2}_a={a}_e_f={e_f}_T_plunge_yr={T_plunge_yr}_z={redshift}"
                sources.append({
                "M": m1,
                "mu": m2,
                "a": a,
                "e_f": e_f,
                "T": T_plunge_yr,
                "z": redshift,
                "repo": source,
                "psd_file": psd_file,
                "model": model,
                "channels": channels,
                "dt": dt,
                "N_montecarlo": Nmonte,
                "device": dev,
                "threshold_SNR": thr_snr,
                })


print("Running the pipeline...")
source_runtimes = {}

# Run the pipeline for each source
# for source in sources:
#     command = (
#         f"python pipeline.py --M {source['M']} --mu {source['mu']} --a {source['a']} "
#         f"--e_f {source['e_f']} --T {source['T']} --z {source['z']} "
#         f"--repo {source['repo']} --psd_file {source['psd_file']} --model {source['model']} --channels {source['channels']} "
#         f"--dt {source['dt']}  --use_gpu --N_montecarlo {source['N_montecarlo']} --device {source['device']}"
#     )
#     if include_foreground:
#         command += " --foreground"
#     if esaorbits:
#         command += " --esaorbits"
#     if tdi2:
#         command += " --tdi2"
    
#     os.system(command)


for source in sources[:1]:
    extra_args = ""
    if include_foreground:
        extra_args += " --foreground"
    if esaorbits:
        extra_args += " --esaorbits"
    if tdi2:
        extra_args += " --tdi2"

    condor_command = (
        f'condor_submit '
        f'-a "M={source["M"]}" '
        f'-a "mu={source["mu"]}" '
        f'-a "a={source["a"]}" '
        f'-a "e_f={source["e_f"]}" '
        f'-a "T={source["T"]}" '
        f'-a "z={source["z"]}" '
        f'-a "repo={source["repo"]}" '
        f'-a "psd_file={source["psd_file"]}" '
        f'-a "model={source["model"]}" '
        f'-a "channels={source["channels"]}" '
        f'-a "dt={source["dt"]}" '
        f'-a "N_montecarlo={source["N_montecarlo"]}" '
        f'-a "device={source["device"]}" '
        f'-a "extra_args={extra_args.strip()}" '
        f'submit_pipeline.submit'
    )
    os.system(condor_command)
