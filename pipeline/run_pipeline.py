import os
import time

# Define the parameters for each source with different repo names
sources = [
    {"M": 1e6, "mu": 1e1, "a": 0.9, "e_f": 0.1, "T": 0.5, "z": 1.0, "repo": "test", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": 3, "device": 3},
    # Add more sources here if needed
]

total_start_time = time.time()

# Run the pipeline for each source
for source in sources:
    start_time = time.time()
    command = (
        f"python pipeline.py --M {source['M']} --mu {source['mu']} --a {source['a']} "
        f"--e_f {source['e_f']} --T {source['T']} --z {source['z']} "
        f"--repo {source['repo']} --psd_file {source['psd_file']} --dt {source['dt']} "
        f"--use_gpu --N_montecarlo {source['N_montecarlo']} --device {source['device']}"
    )
    os.system(command)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime for source {source['repo']}: {elapsed_time:.2f} seconds")

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time
print(f"Total runtime: {total_elapsed_time:.2f} seconds")

# Save total runtime to a file
with open("total_runtime.txt", "w") as f:
    f.write(f"Total runtime: {total_elapsed_time:.2f} seconds")