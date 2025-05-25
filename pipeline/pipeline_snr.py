# nohup python pipeline_snr.py > out_snr.out &
import os
import sys
import numpy as np
# if input is test
if len(sys.argv) > 1 and sys.argv[1] == "test":
    # test mode
    Nmonte = 1
    # device: device to use on GPUs
    dev = 0
    repo_root = "test_snr_"
else:
    # production mode
    Nmonte = 100
    # device: device to use on GPUs
    dev = 0
    repo_root = "production_snr_"

print("Running the pipeline in mode:", repo_root)

#define the psd and response properties
channels = 'AET'
tdi2 = True
model = 'scirdv1'
esaorbits = True
psd_file = "TDI2_AE_psd.npy"
# include_foreground: defines whether to include the confusion noise foreground
include_foreground = True

# source frame parameters
# M: central mass of the binary in solar masses source frame
# mu: secondary mass of the binary in solar masses source frame
# a: dimensionless spin of the central black hole
# e_f: final eccentricity of the binary
# T: observation time in years
# z: redshift of the source
# repo: name of the repository where the results will be saved
# psd_file: name of the file with the power spectral density
# dt: time step in seconds
dt = 5.0
sources = []

m1_values = [1e7, 3162277.6601683795, 1e6, 316227.7660168379, 1e5, 31622.776601683792, 1e4]
m2 = 10.
a_values = [-0.99, 0.0, 0.99]
# e_2yr_values = [1e-4] # Eccentricity does not have a big impact on horizon
e_f = 1e-4
# First find 

for redshift in np.logspace(-2, np.log10(1.5),5):
    for Tobs in [0.5, 1., 2.]:
        for m1 in m1_values:
            for a in a_values:
                source = repo_root + f"m1={m1}_m2={m2}_a={a}_e_f={e_f}_T={Tobs}_z={redshift}"
                sources.append({
                "M": m1 * (1 + redshift),
                "mu": m2 * (1 + redshift),
                "a": a,
                "e_f": e_f,
                "T": Tobs,
                "z": redshift,
                "repo": source,
                "psd_file": psd_file,
                "model": model,
                "channels": channels,
                "dt": dt,
                "N_montecarlo": Nmonte,
                "device": dev,
                "pe": 0,
                })
# save sources to a file
sources_file = "sources_snr.txt"
with open(repo_root + sources_file, "w") as f:
    for source in sources:
        f.write(f"{source}\n")


if len(sys.argv) > 1 and sys.argv[1] == "test":
    sources = sources[:1]  # Only run the first source in test mode

# # Run the pipeline for each source from command
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

# Run the pipeline for each source using condor
for source in sources:
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
        f'-a "calculate_fisher={source["pe"]}" '
        f'-a "extra_args={extra_args.strip()}" '
        f'submit_pipeline.submit'
    )
    os.system(condor_command)
