# nohup python pipeline_pe.py > out_pe.out &
import os
import sys
import json
import numpy as np
# if input is test
if len(sys.argv) > 1 and sys.argv[1] == "test":
    # test mode
    Nmonte = 1
    # device: device to use on GPUs
    dev = 0
    repo_root = "test_pe_"
else:
    # production mode
    Nmonte = 1000
    # device: device to use on GPUs
    dev = 0
    repo_root = "production_pe_"

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
dt = 0.5
Tpl = 0.5  # observation time in years
ef = 1e-8  # final eccentricity of the binary
sources = []
# Load spin, m1, and redshift values from a JSON file
json_file = "requirements_results/snr_redshift_evaluation.json"
with open(json_file, "r") as jf:
    source_data = json.load(jf)

spin = "0.0"
m1_ = np.asarray(source_data[spin]["m1"])
spin_ = np.zeros_like(m1_) + float(spin)
z_ = np.asarray(source_data[spin]["redshift"])
# mask = (m1_ == 1e3) + (m1_ ==  316227.7660168379) + (m1_ == 1e7)
mask = (m1_ > 1e2)
m1_a_z_values = np.column_stack((m1_[mask], spin_[mask], z_[mask]))
for m1, a, z in zip(m1_[mask], spin_[mask], z_[mask]):
    source = repo_root + f"m1={m1}_m2=10._a={a}_e_f=1e-8_T=0.5_z={z}"
    sources.append({"M": m1 * (1 + z),"mu": 10. * (1 + z),"a": a,"e_f": ef,"T": Tpl,"z": z,"repo": source,"psd_file": psd_file,"model": model,"channels": channels,"dt": dt,"N_montecarlo": Nmonte,"device": dev,"pe": 1,})

# missing ./production_pe_m1=1000.0_m2=10._a=0.0_e_f=1e-8_T=0.5_z=0.036521124844052635
# source = repo_root + "m1=1000.0_m2=10._a=0.0_e_f=1e-8_T=0.5_z=0.036521124844052635"
# sources.append({"M": 1000.0 * (1 + 0.036521124844052635),"mu": 10. * (1 + 0.036521124844052635),"a": 0.0,"e_f": 1e-8,"T": 0.5,"z": 0.036521124844052635,"repo": source,"psd_file": psd_file,"model": model,"channels": channels,"dt": dt,"N_montecarlo": Nmonte,"device": dev,"pe": 1,})

spin = "0.99"
m1_ = np.asarray(source_data[spin]["m1"])
spin_ = np.zeros_like(m1_) + float(spin)
z_ = np.asarray(source_data[spin]["redshift"])
# mask = (m1_ == 1e4) + (m1_ == 1e6) + (m1_ == 1e7)
for m1, a, z in zip(m1_[mask], spin_[mask], z_[mask]):
    source = repo_root + f"m1={m1}_m2=10._a={a}_e_f=1e-8_T=0.5_z={z}"
    sources.append({"M": m1 * (1 + z),"mu": 10. * (1 + z),"a": a,"e_f": ef,"T": Tpl,"z": z,"repo": source,"psd_file": psd_file,"model": model,"channels": channels,"dt": dt,"N_montecarlo": Nmonte,"device": dev,"pe": 1,})

spin = "-0.99"
m1_ = np.asarray(source_data[spin]["m1"])
spin_ = np.zeros_like(m1_) + float(spin)
z_ = np.asarray(source_data[spin]["redshift"])
# mask = (m1_ == 1e3) + (m1_ ==  3162277.6601683795) + (m1_ == 1e7)
for m1, a, z in zip(m1_[mask], spin_[mask], z_[mask]):
    source = repo_root + f"m1={m1}_m2=10._a={a}_e_f=1e-8_T=0.5_z={z}"
    sources.append({"M": m1 * (1 + z),"mu": 10. * (1 + z),"a": a,"e_f": ef,"T": Tpl,"z": z,"repo": source,"psd_file": psd_file,"model": model,"channels": channels,"dt": dt,"N_montecarlo": Nmonte,"device": dev,"pe": 1,})

# save sources to a file
sources_file = "sources_pe.txt"
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
#         f"--dt {source['dt']}  --use_gpu --N_montecarlo {source['N_montecarlo']} --device {source['device']} --calculate_fisher {source['pe']} "
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
