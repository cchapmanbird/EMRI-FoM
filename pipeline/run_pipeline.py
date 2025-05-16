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
# decide whether to generate the data for the redshift horizon plot
generate_redshift_horizon = False
# decide whether to plot the redshift horizon plot
plot_redshift_horizon = False

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

#define the psd and response properties
channels = 'AET'
tdi2 = True
model = 'scirdv1'
esaorbits = True
psd_file = "TDI2_AE_psd.npy"
# include_foreground: defines whether to include the confusion noise foreground
include_foreground = True

# horizon settings
T_obs = 2.0 # observation time in years
ntrials = 100 # number of samples over the extrinsic parameters
horizon_outname = "test" # name of the output file
#source frame parameters. In the current setup, the q stated here is ignored as it is not fixed.
qs = 1.e-5
e0s = 0.5
spins = 0.99

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
import json
# open json file with the sources
with open(f"fom_sources_light.json", "r") as json_file:
    source_intr = json.load(json_file)
# breakpoint()
sources = []

# open dictionary with the sources
for source, params in source_intr.items():
    m1 = params["m1"]
    m2 = params["m2"]
    a = params["a"]
    e_f = params["e_f"]
    redshift = params["redshift"]
    T_plunge_yr = params["T_plunge_yr"]
    print("--------------------------------------")
    print(f"Source: {source}")
    print(f"Detector frame m1: {m1}, m2: {m2}")
    print(f"T_plunge_yr: {T_plunge_yr}, redshift: {redshift}")
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
        "threshold_relative_errors": thr_err
    })

# sources = [

#     {"M": 1e6, "mu": 1e1, "a": 0.9, "e_f": 0.01, "T": 0.1, "z": 1.0, "repo": "EMRI", "psd_file": psd_file, "model": model, "channels": channels,  "dt": dt,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
#     {"M": 5e4, "mu": 1e1, "a": 0.0, "e_f": 0.01, "T": 0.1, "z": 0.5, "repo": "LightIMRI", "psd_file": psd_file, "model": model, "channels": channels,  "dt": dt,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
#     {"M": 1e6, "mu": 1e3, "a": 0.9, "e_f": 0.01, "T": 0.1, "z": 1.0, "repo": "HeavyIMRI1", "psd_file": psd_file, "model": model, "channels": channels,  "dt": dt,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
#     {"M": 1e7, "mu": 1e3, "a": 0.9, "e_f": 0.01, "T": 0.1, "z": 1.0, "repo": "HeavyIMRI2", "psd_file": psd_file, "model": model, "channels": channels,  "dt": dt,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
#     {"M": 1e6, "mu": 5e2, "a": 0.9, "e_f": 0.01, "T": 0.1, "z": 1.0, "repo": "HeavyIMRI3", "psd_file": psd_file, "model": model, "channels": channels,  "dt": dt,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err},
# ]
# from mojito_sources import sources_intr
# sources = []
# for src in sources_intr:
#     src_temp = {"M": src["M"],"mu": src["mu"],"a": src["a"],"e_f": src["e_f"],"T": src["T"],"z": src["redshift"],"repo": src["repo"], "psd_file": psd_file, "model": model, "channels": channels,  "dt": dt,  "N_montecarlo": Nmonte, "device": dev, "threshold_SNR": thr_snr, "threshold_relative_errors": thr_err}
#     sources.append(src_temp)
# names of parameters
param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])
# jacobian to obtain source frame Fisher matrix from detector frame Fisher matrix
from common import CosmoInterpolator
cosmo = CosmoInterpolator()

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
            f"--repo {source['repo']} --psd_file {source['psd_file']} --model {source['model']} --channels {source['channels']} "
            f"--dt {source['dt']}  --use_gpu --N_montecarlo {source['N_montecarlo']} --device {source['device']}"
        )
        if include_foreground:
            command += " --foreground"
        if esaorbits:
            command += " --esaorbits"
        if tdi2:
            command += " --tdi2"
        
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

    # load the injected parameters in the source frame
    dict_source = source_intr[source_name]
    # for each of the data_source.files = ['M central black hole mass', 'mu secondary black hole mass', 'a dimensionless central object spin', 'p_f final semi-latus rectum', 'e_f final eccentricity', 'z redshift', 'dist luminosity distance in Gpc', 'T inspiral duration in years']
    # write  in the table the values
    latex_content += rf"""
    \begin{{table}}[h]
    \centering
    \begin{{tabular}}{{|c|c|}}
    \hline
    \textbf{{Parameter}} & \textbf{{Source Frame Value}} & \textbf{{Detector Frame Value}}\\
    \hline
    """
    # loop over data_source.files and data_detector.files
    for param, value in dict_source.items():
        latex_content += rf"{param} & {value}\\"
    latex_content += r"\hline\end{tabular}\end{table}"
    # for i, param in enumerate(param_names):
    #     latex_content += rf"{param} & {injected_params[i]}\\"
    # latex_content += r"\hline\end{tabular}\end{table}"\
    
    # breakpoint()
    latex_content += rf"""
    \begin{{table}}[h]
    \centering
    \begin{{tabular}}{{|c|c|}}
    \hline
    \textbf{{Parameter}} & \textbf{{Detector Frame Value Realization}}\\
    \hline
    """
    for i, param in enumerate(param_names):
        latex_content += rf"{str(param)} & {injected_params[i]}\\"
    latex_content += r"\hline\end{tabular}\end{table}"\
    
    # include SNR plot
    latex_content += rf"""
    \begin{{figure}}
    \centering
    \includegraphics[width=\textwidth]{{{source_name}/snr_histogram.png}}
    \end{{figure}}
    """

    # new section
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

    report_filename = f"{source_name}Assessment.tex"
    with open(report_filename, "w") as f:
        f.write(latex_content)
    print(f"Generated LaTeX report: {report_filename}")
    
    # Compile LaTeX to PDF
    subprocess.run(["pdflatex", "-interaction=nonstopmode", report_filename])
    print(f"Generated PDF report: {source_name}Assessment.pdf")
    # remove aux, log and tex files
    os.remove(f"{source_name}Assessment.aux")
    os.remove(f"{source_name}Assessment.log")
    os.remove(f"{source_name}Assessment.tex")

# Assessment Process
if assess_science_objectives:
    print("Assessing the science objectives...")
    for source in sources:
        source_name = source['repo']
        print(f"Assessing science objectives for {source_name}...")
        filelist = sorted(glob.glob(f"{source_name}/*/results.npz"))
        total_results = {nn: [] for nn in ['cov', 'snr', 'fisher_params', 'errors', 'relative_errors']}
        for file in filelist:
            print(f"Processing file: {file}")
            results = np.load(file)
            # source_frame_data = np.load(f"{source_name}/source_frame_data.npz")
            redshift = source_intr[source_name]["redshift"]
            M_source = source_intr[source_name]["m1_source"]
            mu_source = source_intr[source_name]["m2_source"]
            J = cosmo.jacobian(M_source, mu_source, redshift)
            for el in total_results.keys():
                if el == 'cov':
                    Gamma = np.linalg.inv(results[el])
                    total_results[el].append(np.linalg.inv(J.T @ Gamma @ J))
                    # total_results[el].append(results[el])
                if el == 'fisher_params':
                    source_frame_par = results[el]
                    source_frame_par[0] = source_frame_par[0] / (1+redshift)
                    source_frame_par[1] = source_frame_par[1] / (1+redshift)
                    total_results[el].append(source_frame_par)
                else:
                    total_results[el].append(results[el])
        
        mean_snr = np.mean(total_results['snr'])
        # plot SNR histogram
        plt.figure()
        plt.hist(np.log10(total_results['snr']), bins=30)
        plt.axvline(np.log10(source['threshold_SNR'][0]), color='r', linestyle='--', label=f'Threshold {source["threshold_SNR"][0]}')
        plt.axvline(np.log10(source['threshold_SNR'][1]), color='orange', linestyle='--', label=f'Threshold {source["threshold_SNR"][1]}')
        plt.axvline(np.log10(source['threshold_SNR'][2]), color='green', linestyle='--', label=f'Threshold {source["threshold_SNR"][2]}')
        plt.xlabel('Log10 SNR')
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig(f"{source_name}/snr_histogram.png")
        plt.close()
        # save snr distribution in folder
        np.savez(f"{source_name}/snr_distribution.npz", snr=total_results['snr'])

        par_vals = np.array(total_results['fisher_params'])
        errors = np.asarray([np.diag(el) for el in total_results['cov']])**0.5 # / total_results['fisher_params'][0]
        # update relative errors
        source_frame_rel_errors = np.asarray([np.diag(el) for el in total_results['cov']])**0.5 / total_results['fisher_params'][0]
        det_frame_rel_err = np.asarray(total_results['relative_errors'])
        mean_relative_errors = np.mean(np.asarray(source_frame_rel_errors),axis=0)
        errors = np.asarray([np.diag(el) for el in total_results['cov']])**0.5 # / total_results['fisher_params'][0]
        # mask parameter values with zeros
        mask_zeros = (par_vals[0] == 0)
        # create a histogram of the relative errors
        rel_err = np.asarray(source_frame_rel_errors)
        rel_err[:,mask_zeros] = errors[:,mask_zeros]

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
        
        Sigma = np.asarray(total_results['cov'])[:,6:8, 6:8]
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
                # if par_vals[0][i] != 0:
                plt.figure()
                # add title with median and std
                plt.title(f'Median: {np.median(rel_err[:,i]):.2e}, Std: {np.std(rel_err[:,i]):.2e}')
                plt.hist(np.log10(rel_err[:,i]), bins=30, label='Source Frame', alpha=0.7, density=True)
                # check if nans are present
                if np.sum(np.isinf(det_frame_rel_err[:,i]))==0:
                    plt.hist(np.log10(det_frame_rel_err[:,i]), bins=30, label='Detector Frame', alpha=0.7, density=True)
                plt.axvline(np.log10(threshold), color='r', linestyle='--', label='Threshold')
                plt.xlabel(f'Log10 Relative Error {param}')
                # plt.ylabel('Counts')
                plt.legend()
                plt.savefig(f"{source_name}/relative_errors_histogram_{param}.png")
                plt.close()
                np.savez(f"{source_name}/relative_errors_histogram_{param}.npz", relative_error=rel_err[:,i])
                
            if param == 'qS':
                status = "PASS" if np.median(err_sky_loc) < threshold else "FAIL"
                error_statuses.append(("Sky Localization", status, np.median(err_sky_loc), threshold))
                plt.figure()
                plt.title(f'Median: {np.median(err_sky_loc):.2e}, Std: {np.std(err_sky_loc):.2e}')
                plt.hist(err_sky_loc, bins=30, label='Source Frame')
                plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
                plt.xlabel(f'Relative Error Sky Localization')
                # plt.ylabel('Counts')
                plt.savefig(f"{source_name}/relative_errors_histogram_skyloc.png")
                plt.close()
                np.savez(f"{source_name}/relative_errors_histogram_skyloc.npz", relative_error=err_sky_loc)
            
            
        
        # Generate LaTeX report and compile to PDF
        generate_latex_report(source_name, snr_status, mean_snr, error_statuses, total_results['fisher_params'][0])

if generate_redshift_horizon:
    print("Generating data for redshift horizon plot")
    start_time = time.time()
    command = (
        f"python horizon/produce_data.py --dev {dev} "
        f"--model {model} --channels {channels} "
        f"--dt {dt} --Tobs {T_obs} --outname {horizon_outname} --avg_n {ntrials} "
        f"--traj kerr --wf kerr --qs {qs} --e0s {e0s} --spins {spins} --grids q M"
    )

    if include_foreground:
        command += " --foreground"
    if esaorbits:
        command += " --esaorbits"
    if tdi2:
        command += " --tdi2"
    
    os.system(command)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime for the horizon data production: {elapsed_time:.2f} seconds")

if plot_redshift_horizon:
    command = (
        f"python horizon/plot_data.py -Tobs {T_obs} -q {qs} -e0 {e0s} -spin {spins} -zaxis q -base {horizon_outname} -interp -cpal inferno_r"
    )
    os.system(command)
    print(f"Plotted the redshift horizon plot")

