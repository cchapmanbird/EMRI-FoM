import os
import time
import glob
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import io
import healpy as hp

# decide whether to run the full pipeline and generate the results
run_pipeline = False
# decide whether to assess the science objectives
assess_science_objectives = True

# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [10.0, 15., 20.]
# threshold_relative_errors: threshold on relative errors for the science objectives    
#           M    mu    a     p0    e0   dist   qS   phiS qK   phiK Phi_phi0 Phi_r0
thr_err = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-1, 10., 10., 10., 10., 10.,     10.]
# qs is used as sky localization threshold
# p0, phi0, theta0 are not used in the threshold

# device: device to use on GPUs
dev = 0
# defines the number of montecarlo runs over phases and sky locations
# N_montecarlo: number of montecarlo runs over phases and sky locations
Nmonte = 100

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

sources = [
    {"M": 1e6, "mu": 1e1, "a": 0.9, "e_f": 0.2, "T": 1.0, "z": 1.0, "repo": "Eccentric", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    {"M": 1e6, "mu": 1e1, "a": 0.9, "e_f": 0.01, "T": 1.0, "z": 1.0, "repo": "Circular", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    # {"M": 1e7, "mu": 1e1, "a": 0.9, "e_f": 0.2, "T": 1.0, "z": 0.5, "repo": "HighMass", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    # {"M": 1e5, "mu": 1e1, "a": 0.9, "e_f": 0.2, "T": 1.0, "z": 0.5, "repo": "LowMass", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    # {"M": 0.5e6, "mu": 1e1, "a": 0.9, "e_f": 0.1, "T": 1.0, "z": 1.0, "repo": "IMRI", "psd_file": "TDI2_AE_psd.npy", "dt": 10.0,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
    # Add more sources here if needed
]

# names of parameters
param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])
popinds = []
popinds.append(5)
popinds.append(12)
param_names = np.delete(param_names, popinds).tolist()


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


# Function to generate LaTeX report and compile to PDF
def generate_latex_report(source_name, snr_status, snr_value, error_statuses, injected_params):

    latex_content = rf"""
    \documentclass{{article}}
    \usepackage{{graphicx}}
    \usepackage{{booktabs}}
    \begin{{document}}
    \title{{Science Objectives Assessment for {source_name}}}
    \author{{Automated Report}}
    \date{{\today}}
    \maketitle
    \section{{Summary}}
    \textbf{{Source:}} {source_name}\\
    \textbf{{SNR Status:}} {snr_status} (Mean SNR = {snr_value:.2f})\\
    """

    # add table of injected parameters
    latex_content += rf"""
    \begin{{table}}[h]
    \centering
    \begin{{tabular}}{{|c|c|}}
    \hline
    \textbf{{Parameter}} & \textbf{{Detector Frame Value}}\\
    \hline
    """
    for i, param in enumerate(param_names):
        latex_content += rf"{param} & {injected_params[i]}\\"
    latex_content += r"\hline\end{tabular}\end{table}"
    latex_content += r"\section{{Parameter Relative Errors}}"

    for param, status, value, threshold in error_statuses:
        latex_content += rf"\textbf{{{param}:}} {status} (Mean Error = {value:.2e}, Threshold = {threshold:.2e})\\\n"

    # add plots of all the relative errors in relative_errors_histogram_{param}.png
    for param in ['M','mu','a','e0','dist','skyloc']:
        latex_content += rf"""
        \begin{{figure}}
        \centering
        \includegraphics[width=\textwidth]{{{source_name}/relative_errors_histogram_{param}.png}}
        \end{{figure}}
        """
    # create a section with the covariance ellipes plot and the waveform
    latex_content += rf"""
    \begin{{figure}}
    \centering
    \includegraphics[width=\textwidth]{{{source_name}/realization_0/covariance_ellipse_plot.png}}
    \includegraphics[width=\textwidth]{{{source_name}/waveform.png}}
    \end{{figure}}
    """

    # end of the document
    latex_content += "\\end{document}"

    report_filename = f"{source_name}_assessment.tex"
    with open(report_filename, "w") as f:
        f.write(latex_content)
    print(f"Generated LaTeX report: {report_filename}")
    
    # Compile LaTeX to PDF
    subprocess.run(["pdflatex", "-interaction=nonstopmode", report_filename])
    print(f"Generated PDF report: {source_name}_assessment.pdf")
    # remove aux, log and tex files
    os.remove(f"{source_name}_assessment.aux")
    os.remove(f"{source_name}_assessment.log")
    os.remove(f"{source_name}_assessment.tex")

# Assessment Process
if assess_science_objectives:
    print("Assessing the science objectives...")
    for source in sources:
        source_name = source['repo']
        print(f"Assessing science objectives for {source_name}...")
        filelist = glob.glob(f"{source_name}/*/*.npz")
        total_results = {nn: [] for nn in ['cov', 'snr', 'fisher_params', 'errors', 'relative_errors']}
        
        for file in filelist:
            results = np.load(file)
            for el in total_results.keys():
                total_results[el].append(results[el])
        
        mean_snr = np.mean(total_results['snr'])
        # mean_relative_errors = np.diag(np.mean(total_results['cov'], axis=0)) / total_results['fisher_params'][0]
        par_vals = np.array(total_results['fisher_params'])
        mean_relative_errors = np.mean(np.asarray(total_results['relative_errors']),axis=0)
        # plot a mollview of the angles theta=par_vals[:,6] phi=par_vals[:,7] with colormap from the error
        # nside = 16
        # npix = hp.nside2npix(nside)
        # sky_map = np.zeros(npix)
        # theta = par_vals[:,6]
        # phi = par_vals[:,7]
        # pixels = hp.ang2pix(nside, theta, phi)
        # for pix, err in zip(pixels, np.asarray(total_results['snr'])):
        #     sky_map[pix] += err
        # sky_map /= np.bincount(pixels, minlength=npix)
        # hp.mollview(sky_map, title=f'Mollview of SNR', unit='Relative Error')
        # plt.savefig(f"{source_name}/mollview_snr_sky.png")
        # plt.close()

        # npix = hp.nside2npix(nside)
        # sky_map = np.zeros(npix)
        # theta = par_vals[:,8]
        # phi = par_vals[:,9]
        # pixels = hp.ang2pix(nside, theta, phi)
        # for pix, err in zip(pixels, np.asarray(total_results['snr'])):
        #     sky_map[pix] += err
        # sky_map /= np.bincount(pixels, minlength=npix)
        # hp.mollview(sky_map, title=f'Mollview of SNR', unit='Relative Error')
        # plt.savefig(f"{source_name}/mollview_snr_orientation.png")
        # plt.close()
        
        # create a histogram of the relative errors
        rel_err = np.asarray(total_results['relative_errors'])

        Sigma = np.mean(total_results['cov'], axis=0)[6:8, 6:8]
        thetaS, phiS = total_results['fisher_params'][0][6:8]
        err_sky_loc = 2 * np.pi * np.sin(thetaS) * np.sqrt(np.linalg.det(Sigma)) * (180.0 / np.pi) ** 2
        
        # Evaluate SNR
        snr_status = ""
        if mean_snr < source['threshold_SNR'][0]:
            snr_status = rf"FAIL (SNR $<$ {source['threshold_SNR'][0]})"
        elif mean_snr < source['threshold_SNR'][1]:
            snr_status = rf"WARNING (SNR $<$ {source['threshold_SNR'][1]})"
        elif mean_snr < source['threshold_SNR'][2]:
            snr_status = rf"GOOD (SNR $<$ {source['threshold_SNR'][2]})"
        else:
            snr_status = rf"PASS (SNR $>$ {source['threshold_SNR'][2]})"
        
        # Evaluate Relative Errors
        error_statuses = []
        for i, err_value in enumerate(mean_relative_errors):
            param = param_names[i]
            threshold = source['threshold_relative_errors'][i]
            if param in ['M', 'mu', 'a', 'e0', 'dist']:
                status = "PASS" if err_value < threshold else "FAIL"
                error_statuses.append((param, status, err_value, threshold))
                plt.figure()
                plt.hist(np.log10(rel_err[:,i]), bins=30, label='Montecarlo Runs')
                plt.axvline(np.log10(threshold), color='r', linestyle='--', label='Threshold')
                plt.xlabel(f'Log10 Relative Error {param}')
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(f"{source_name}/relative_errors_histogram_{param}.png")
                plt.close()
                
            if param == 'qS':
                status = "PASS" if err_sky_loc < threshold else "FAIL"
                error_statuses.append(("Sky Localization", status, err_sky_loc, threshold))
                plt.figure()
                plt.hist(err_sky_loc, bins=30, label='Montecarlo Runs')
                plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
                plt.xlabel(f'Relative Error Sky Localization')
                plt.ylabel('Counts')
                plt.savefig(f"{source_name}/relative_errors_histogram_skyloc.png")
                plt.close()
            
            
        
        # Generate LaTeX report and compile to PDF
        generate_latex_report(source_name, snr_status, mean_snr, error_statuses, total_results['fisher_params'][0])
