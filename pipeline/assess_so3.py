import os
import time
import glob
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import io
import healpy as hp
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [20.0, 25., 30.]
# threshold_relative_errors: threshold on relative errors for the science objectives    
#           M    mu    a     p0    e0   dist   qS   phiS qK   phiK Phi_phi0 Phi_r0
thr_err = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-1, 10., 10., 10., 10., 10.,     10.]

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


list_folders = sorted(glob.glob("./m1*"))
print(list_folders)
list_results = []
# Assessment Process
for source in list_folders:
    print(f"Assessing science objectives for {source}...")
    # z_red = float(source.split("z=")[-1])
    m1_str = float(source.split("m1=")[-1].split("_")[0])
    # if m1_str not in [1e7, 1e6, 1e5, 1e4]:
    #     print(f"Skipping source {source}")
    #     continue
    Tpl = float(source.split("yr=")[-1].split("_")[0])
    redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
    detector_params = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["parameters"]
    source_params = detector_params.copy()
    source_params[0] = source_params[0] / (1 + redshift)
    source_params[1] = source_params[1] / (1 + redshift)
    # SNR assessment
    snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
    print(f"Mean SNR: {np.mean(snr_list)}")
    list_results.append({"m1":source_params[0], "m2":source_params[1], "snr":snr_list, "redshift":redshift, "dist":source_params[6], "Tpl": Tpl})
    # plot SNR histogram
    plt.figure()
    plt.hist(np.log10(snr_list), bins=30)
    plt.axvline(np.log10(thr_snr[0]), color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
    plt.axvline(np.log10(thr_snr[1]), color='orange', linestyle='--', label=f'Threshold {thr_snr[1]}')
    plt.axvline(np.log10(thr_snr[2]), color='green', linestyle='--', label=f'Threshold {thr_snr[2]}')
    plt.xlabel('Log10 SNR')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(f"{source}/snr_histogram.png")
    plt.close()
    # save snr distribution in folder
    np.savez(f"{source}/snr_distribution.npz", snr=snr_list)

    # parameter estimation
    # detector_params = np.load(sorted(glob.glob(f"{source}/*/results.npz"))[0])["parameters"]


# Collect m1, redshift, and mean SNR for each source
m1_vals = np.asarray([np.log10(res["m1"]) for res in list_results if res["Tpl"] < 1])
redshift_vals = np.asarray([res["redshift"] for res in list_results if res["Tpl"] < 1])
mean_snr_vals = np.asarray([np.mean(res["snr"]) for res in list_results if res["Tpl"] < 1])
# For each unique m1, interpolate SNR vs redshift and find redshift where SNR=20
snr_thresholds = [20.0, 25.0, 30.0]
results_redshift_at_snr = {thr: {} for thr in snr_thresholds}

unique_m1 = np.unique(m1_vals)
for snr_threshold in snr_thresholds:
    for m1 in unique_m1:
        mask = m1_vals == m1
        z = redshift_vals[mask]
        snr = mean_snr_vals[mask]
        # Sort by redshift for interpolation
        sort_idx = np.argsort(z)
        z_sorted = z[sort_idx]
        snr_sorted = snr[sort_idx]
        # Only interpolate if SNR crosses the threshold
        if np.any(snr_sorted >= snr_threshold) and np.any(snr_sorted <= snr_threshold):
            try:
                z_at_snr = np.interp(np.log10(snr_threshold), np.log10(snr_sorted[::-1]), z_sorted[::-1])
                results_redshift_at_snr[snr_threshold][m1] = z_at_snr
                print(f"m1={10**m1:.1e}, redshift at SNR={snr_threshold}: {z_at_snr:.3f}")
            except Exception as e:
                print(f"Interpolation failed for m1={10**m1:.1e}, SNR={snr_threshold}: {e}")
        else:
            print(f"m1={10**m1:.1e}: SNR does not cross threshold {snr_threshold}")

# Plot redshift at each SNR threshold vs m1
plt.figure()
for snr_threshold in snr_thresholds:
    if results_redshift_at_snr[snr_threshold]:
        m1_plot = np.array(list(results_redshift_at_snr[snr_threshold].keys()))
        z_plot = np.array(list(results_redshift_at_snr[snr_threshold].values()))
        plt.plot(m1_plot, z_plot, 'o-', label=f'Redshift at SNR={snr_threshold}')
plt.xlabel(r'$\log_{10} m_1$')
plt.ylabel('Redshift at SNR threshold')
plt.title('Redshift where SNR threshold is reached vs Mass')
plt.legend()
plt.tight_layout()
plt.savefig("redshift_at_snr_thresholds_vs_mass.png")
plt.close()
plt.figure()
# Define a colormap with 4 colors: red, orange, green, blue
# Interpolate onto a grid for smoothing

# Define grid
m1_grid = np.linspace(m1_vals.min(), m1_vals.max(), 20)
redshift_grid = np.linspace(redshift_vals.min(), redshift_vals.max(), 10)
M1, Z = np.meshgrid(m1_grid, redshift_grid)

# Interpolate mean_snr_vals onto the grid
snr_grid = griddata(
    (m1_vals, redshift_vals), mean_snr_vals, (M1, Z), method='linear'
)

levels = 5
# Plot smoothed contour
sc = plt.contourf(M1, Z, snr_grid, levels=levels, cmap='viridis')
# plt.xscale('log')
plt.xlabel(r'$m_1 [M_\odot]$')
plt.ylabel('Redshift')
cbar = plt.colorbar(sc)
cbar.set_label(r'SNR')
plt.ylim(0, 1.)
plt.tight_layout()
plt.savefig("Mass_redshift_snr_scatter.png")
plt.close()

plt.figure()
sc = plt.scatter(redshift_vals, mean_snr_vals, c=m1_vals, cmap='viridis', s=60)
plt.axhline(thr_snr[0], color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
plt.yscale('log')
plt.xlabel('Redshift')
plt.ylabel('SNR')
cbar = plt.colorbar(sc)
cbar.set_label(r'$\log_{10} m_1$')
plt.tight_layout()
plt.savefig("SNR_vs_redshift.png")
plt.close()

plt.figure()
sc = plt.scatter(m1_vals, mean_snr_vals, c=redshift_vals, cmap='viridis', s=60)
plt.axhline(thr_snr[0], color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
plt.yscale('log')
plt.xlabel(r'$\log_{10} m_1$')
plt.ylabel('SNR')
cbar = plt.colorbar(sc)
cbar.set_label("Redshift")
plt.tight_layout()
plt.savefig("SNR_vs_mass.png")
plt.close()
