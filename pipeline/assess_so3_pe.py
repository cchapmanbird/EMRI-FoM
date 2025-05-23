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
import os

# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [20.0, 25., 30.]

list_folders = sorted(glob.glob("./production_pe_m*"))
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
    # pe assessment
    source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
    fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
    fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
    fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
    relative_error = np.asarray([np.sqrt(np.diag(source_cov[ii]))/fish_params[ii] for ii in range(len(fish_params))])
    err_sky_loc = np.asarray([np.load(el)["err_sky_loc"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
    names = np.asarray([np.load(el)["names"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])[0]
    names[6] = "Omega"
    names = names[:7]
    relative_error[:, 6] = err_sky_loc
    for ii, el in enumerate(names):
        error = relative_error[:, ii]
        plt.figure()
        plt.hist(np.log10(error), bins=30)
        # plt.axvline(np.log10(thr_snr[0]), color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
        # plt.axvline(np.log10(thr_snr[1]), color='orange', linestyle='--', label=f'Threshold {thr_snr[1]}')
        # plt.axvline(np.log10(thr_snr[2]), color='green', linestyle='--', label=f'Threshold {thr_snr[2]}')
        plt.xlabel('Log10 relative error ' + el)
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig(f"{source}/{el}_histogram.png")
        plt.close()
        # save snr distribution in folder
        np.savez(f"{source}/{el}_distribution.npz", error=error)

