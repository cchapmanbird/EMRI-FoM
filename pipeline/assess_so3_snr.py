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
import matplotlib.colors as mcolors
from common import CosmoInterpolator
import h5py
cosmo = CosmoInterpolator()
import json
import glob
from matplotlib.lines import Line2D


label_fontsize = 18
tick_fontsize = 18
title_fontsize = 18

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["figure.figsize"] = (4, 3)
# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [20.0, 25., 30.]

list_folders = sorted(glob.glob("./production_snr_m*"))
print(list_folders)
list_results = []
h5_path = "snr_results.h5"
# check if the HDF5 file already exists
if os.path.exists(h5_path):
    print(f"HDF5 file {h5_path} already exists.")
else:
    with h5py.File(h5_path, "w") as h5f:
        # Assessment Process
        for source in list_folders:
            print(f"Assessing science objectives for {source}...")
            m1_str = float(source.split("m1=")[-1].split("_")[0])
            Tobs = float(source.split("T=")[-1].split("_")[0])
            signed_spin = float(source.split("a=")[-1].split("_")[0])
            snr_file = sorted(glob.glob(f"{source}/*/snr.npz"))[0]
            data = np.load(snr_file)
            redshift = data["redshift"]
            detector_params = data["parameters"]
            source_params = detector_params.copy()
            source_params[0] = source_params[0] / (1 + redshift)
            source_params[1] = source_params[1] / (1 + redshift)
            snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
            print(f"Mean SNR: {np.mean(snr_list)}")
            result = {
                "m1": source_params[0],
                "m2": source_params[1],
                "a": signed_spin,
                "p0": source_params[3],
                "e0": source_params[4],
                "snr": snr_list,
                "redshift": redshift,
                "dist": source_params[6],
                "Tobs": Tobs
            }
            list_results.append(result)
            # Store in HDF5
            grp = h5f.create_group(source)
            for k, v in result.items():
                grp.create_dataset(k, data=v)
            # plot SNR histogram
            plt.figure()
            plt.hist(np.log10(snr_list), bins=30)
            plt.axvline(np.log10(thr_snr[0]), color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
            plt.axvline(np.log10(thr_snr[1]), color='orange', linestyle='--', label=f'Threshold {thr_snr[1]}')
            plt.axvline(np.log10(thr_snr[2]), color='green', linestyle='--', label=f'Threshold {thr_snr[2]}')
            plt.xlabel('Log10 SNR')
            plt.ylabel('Counts')
            plt.legend()
            plt.savefig(f"{source}/snr_histogram.png",dpi=300)
            plt.close()
            # save snr distribution in folder
            np.savez(f"{source}/snr_distribution.npz", snr=snr_list)
    print(f"Results saved in {h5_path}")