import os
import time
import glob
import numpy as np
# Define the parameters in the source frame for each source with different repo names
thr_snr = [10.0, 15., 20.]
# threshold on relative errors for each parameter, for sky localization in degrees sqaured for variable qS and phiS
#           M    mu    a     p0    e0   dist   qS   phiS qK   phiK Phi_phi0 Phi_r0
thr_err = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-1, 10., 10., 10., 10., 10.,     10.]
param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])
popinds = []
popinds.append(5)
popinds.append(12)
param_names = np.delete(param_names, popinds).tolist()

sources = [
    {"M": 1e6, "mu": 1e1, "a": 0.9, "e_f": 0.1, "T": 0.5, "z": 1.0, "repo": "eccentric", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": 10, "device": 3, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    {"M": 1e6, "mu": 1e1, "a": 0.9, "e_f": 0.0001, "T": 0.5, "z": 1.0, "repo": "circular", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": 10, "device": 3, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    # {"M": 0.5e6, "mu": 1e1, "a": 0.9, "e_f": 0.1, "T": 0.5, "z": 1.0, "repo": "imri", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": 10, "device": 3, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    # Add more sources here if needed
]

run_pipeline = True
assess_science_objectives = True

if run_pipeline:
    print("Running the pipeline...")
    total_start_time = time.time()
    source_runtimes = {}

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
        source_runtimes[source['repo']] = elapsed_time
        print(f"Runtime for source {source['repo']}: {elapsed_time:.2f} seconds")

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total runtime: {total_elapsed_time:.2f} seconds")

    # Save total runtime and individual source runtimes to a file
    with open("total_runtime_pipeline.txt", "w") as f:
        f.write(f"Total runtime: {total_elapsed_time:.2f} seconds\n")
        for repo, runtime in source_runtimes.items():
            f.write(f"Runtime for source {repo}: {runtime:.2f} seconds\n")

names = ['cov', 'snr', 'fisher_params', 'errors', 'relative_errors']
total_results = {source['repo']: {nn: [] for nn in names} for source in sources}

if assess_science_objectives:
    print("Assessing the science objectives...")

    # assess science objectives
    for source in sources:
        print('***********************************')
        print(f"Assessing science objectives for source {source['repo']}...")
        filelist = glob.glob(f"{source['repo']}/*/*.npz")
        
        for file in filelist:
            results = np.load(file)
            for el in names:
                total_results[source['repo']][el].append(results[el])
        
        # assess the science objectives for each source
        mean_snr = np.mean(total_results[source['repo']]['snr'])
        mean_relative_errors = np.diag(np.mean(total_results[source['repo']]['cov'],axis=0))/total_results[source['repo']]['fisher_params'][0]
        # sky loc error as estimated in https://arxiv.org/pdf/2102.01708.pdf sec3.2
        Sigma = np.mean(total_results[source['repo']]['cov'],axis=0)[6:8,6:8]
        thetaS, phiS = total_results[source['repo']]['fisher_params'][0][6:8]
        err_sky_loc = 2*np.pi*np.sin(thetaS)*np.sqrt(np.linalg.det(Sigma)) * (180.0/(np.pi))**2 
        print('-----------------------------------')
        print("Science objectives based on SNR")
        # assess the science objectives for each source by finding in which SNR range based on three thresholds of SNR
        thrershold_snr = source['threshold_SNR']
        # find in which SNR range based on three thresholds of SNR the source is
        if mean_snr < thrershold_snr[0]:
            print(f"Source {source['repo']} has SNR < 10, nooo let's save the science objectives! Mean SNR=", mean_snr)
        elif mean_snr < thrershold_snr[1]:
            print(f"Source {source['repo']} has SNR < 15, we can do better come on ESA Mean SNR=", mean_snr)
        elif mean_snr < thrershold_snr[2]:
            print(f"Source {source['repo']} has SNR < 20, we are doing good Mean SNR=", mean_snr)
        else:
            print(f"Source {source['repo']} has SNR > 20, we are doing great! Mean SNR=", mean_snr)
        print('-----------------------------------')
        print("Science objectives based on relative errors")
        # assess the science objectives for each source by finding in which relative errors range based on three thresholds of relative errors
        thrershold_relative_errors = source['threshold_relative_errors']
        # find in which relative errors range based on three thresholds of relative errors the source is
        for i,el in enumerate(mean_relative_errors):
            if param_names[i] in ['M','mu','a','e0','dist']:
                print(f"Parameter {param_names[i]}")
                if el < thrershold_relative_errors[i]:
                    print(f"Source {source['repo']} has relative errors < {thrershold_relative_errors[i]}, we are doing great! Mean relative error=", el)
                else:
                    print(f"Source {source['repo']} has relative errors > {thrershold_relative_errors[i]}, we can do better come on ESA. Mean relative error=", el)
            if param_names[i] == 'qS':
                print(f"Sky localization")
                if err_sky_loc < thrershold_relative_errors[i]:
                    print(f"Source {source['repo']} has relative errors < {thrershold_relative_errors[i]}, we are doing great! Mean relative error=", err_sky_loc)
                else:
                    print(f"Source {source['repo']} has relative errors > {thrershold_relative_errors[i]}, we can do better come on ESA. Mean relative error=", err_sky_loc)
        print('***********************************')

