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
import h5py


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

list_folders = sorted(glob.glob("./production_pe_m*"))
print(list_folders)
list_results = []
h5_path = "pe_results.h5"
# check if the HDF5 file already exists
if os.path.exists(h5_path):
    print(f"HDF5 file {h5_path} already exists.")
else:
    with h5py.File(h5_path, "w") as h5f:
        # Assessment Process
        for source in list_folders:
            print(f"Assessing science objectives for {source}...")
            m1_str = float(source.split("m1=")[-1].split("_")[0])
            Tpl = float(source.split("T=")[-1].split("_")[0])
            redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
            detector_params = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["parameters"]
            source_params = detector_params.copy()
            source_params[0] = source_params[0] / (1 + redshift)
            source_params[1] = source_params[1] / (1 + redshift)
            # if (source_params[0] > 5e6)or(source_params[0] < 5e3)or(float(detector_params[2])!=1e-7):
            #     print(f"Skipping {source} because m1={source_params[0]} > 1e6")
            #     continue
            # pe assessment
            source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            detector_cov = np.asarray([np.load(el)["cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
            fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
            lum_dist = fish_params[:,5]
            sky_loc = fish_params[:,6:8]
            eccentricity = fish_params[:, 4]
            assert np.sum(np.sum(np.diff(fish_params,axis=0),axis=0)[:6])==0.0

            # SNR assessment
            snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])

            plt.figure()
            plt.hist(snr_list, bins=30)
            plt.xlabel('SNR')
            plt.ylabel('Counts')
            plt.savefig(f"{source}/snr_histogram.png",dpi=300)
            plt.figure()
            nside = 12
            npix = hp.nside2npix(nside)
            sky_map = np.zeros(npix)
            theta = sky_loc[:, 0]
            phi = sky_loc[:, 1]
            pixels = hp.ang2pix(nside, theta, phi)
            for i, pix in enumerate(pixels):
                sky_map[pix] += snr_list[i]
            counts = np.bincount(pixels, minlength=npix)
            counts[counts == 0] = 1  # avoid division by zero
            sky_map = sky_map / counts
            hp.mollview(sky_map, title=f"SNR across sky", unit="SNR", cmap="viridis")
            plt.savefig(f"{source}/snr_sky.png",dpi=300)
            plt.close()

            measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
            detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
            err_sky_loc = np.asarray([np.load(el)["err_sky_loc"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            names = np.asarray([np.load(el)["names"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])[0]
            # add sky location to names
            names[6] = "Omega"
            # use inclination names[8] as a proxy for iota
            names[7] = "iota"
            # define up to sky location
            names = names[:8]
            measurement_precision[:, 6] = err_sky_loc
            detector_measurement_precision[:, 6] = err_sky_loc  # Assuming sky loc error is same for both
            # inclination
            measurement_precision[:, 7] = measurement_precision[:, 8]
            detector_measurement_precision[:, 7] = detector_measurement_precision[:, 8]

            # Prepare result dict for HDF5
            result = {
                "m1": source_params[0],
                "m2": source_params[1],
                "Tpl": Tpl,
                "redshift": redshift,
                "lum_dist": lum_dist,
                "eccentricity": eccentricity,
                "snr": snr_list,
                "sky_loc": sky_loc
            }
            list_results.append(result)
            # Store in HDF5
            grp = h5f.create_group(source)
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    grp.create_dataset(k, data=v)
                else:
                    grp.attrs[k] = v

            for ii, el in enumerate(names):
                error_source = measurement_precision[:, ii]
                error_detector = detector_measurement_precision[:, ii]
                if (el == "M") or (el == "mu"):
                    error_source = error_source / source_params[ii]
                    error_detector = error_detector / detector_params[ii]
                    xlabel = 'Relative error ' + el
                else:
                    xlabel = 'Absolute error ' + el

                # Histogram plot
                plt.figure()
                plt.hist(error_source, bins=30, alpha=0.6, label='Source frame')
                plt.hist(error_detector, bins=30, alpha=0.6, label='Detector frame')
                plt.xlabel(xlabel)
                plt.ylabel('Counts')
                plt.legend()
                plt.savefig(f"{source}/{el}_histogram.png",dpi=300)
                plt.close()

                # SNR vs error plot
                plt.figure()
                plt.plot(snr_list, error_source, 'o', label='Source frame')
                plt.plot(snr_list, error_detector, 'x', label='Detector frame')
                snr_vec = np.linspace(np.min(snr_list), np.max(snr_list), len(lum_dist))
                if el == "dist":
                    plt.plot(snr_vec, lum_dist / snr_vec, 'r--', label='d/SNR')
                else:
                    plt.plot(snr_vec, 1 / snr_vec, 'r--', label='1/SNR')
                plt.xlabel('SNR')
                plt.ylabel(xlabel)
                plt.legend()
                plt.savefig(f"{source}/snr_{el}.png",dpi=300)
                plt.close()

                # plot of SNR across the sky
                plt.figure()
                nside = 8
                npix = hp.nside2npix(nside)
                sky_map = np.zeros(npix)
                theta = sky_loc[:, 0]
                phi = sky_loc[:, 1]
                pixels = hp.ang2pix(nside, theta, phi)
                # Accumulate the sum of errors in each pixel
                for i, pix in enumerate(pixels):
                    sky_map[pix] += error_source[i]
                counts = np.bincount(pixels, minlength=npix)
                counts[counts == 0] = 1  # avoid division by zero
                sky_map = sky_map / counts  # average error per pixel
                hp.mollview(sky_map, title=f"Precision error across sky for {el}", unit="Error", cmap="viridis")
                plt.savefig(f"{source}/precision_error_sky_{el}.png",dpi=300)
                plt.close()

                # Save distributions
                np.savez(f"{source}/{el}_distribution.npz", error_source=error_source, error_detector=error_detector)
                # Save errors in HDF5
                err_grp = grp.create_group(f"errors_{el}")
                err_grp.create_dataset("error_source", data=error_source)
                err_grp.create_dataset("error_detector", data=error_detector)

# Plot mean and std precision of parameters as a function of m1
import matplotlib.pyplot as plt

with h5py.File(h5_path, "r") as h5f:
    m1_list = []
    param_means = {}
    param_stds = {}
    param_names = None

    # Gather all parameter names from the first group
    for group in h5f:
        for err_key in h5f[group]:
            if err_key.startswith("errors_"):
                if param_names is None:
                    param_names = []
                param_names.append(err_key.replace("errors_", ""))
        break

    # Initialize storage
    for pname in param_names:
        param_means[pname] = []
        param_stds[pname] = []

    # Loop through groups
    for group in h5f:
        m1 = h5f[group].attrs["m1"] if "m1" in h5f[group].attrs else h5f[group]["m1"][()]
        m1_list.append(m1)
        for pname in param_names:
            err_grp = h5f[group][f"errors_{pname}"]
            error_source = err_grp["error_source"][()]
            param_means[pname].append(np.mean(error_source))
            param_stds[pname].append(np.std(error_source))

    m1_arr = np.array(m1_list)
    sort_idx = np.argsort(m1_arr)
    m1_arr = m1_arr[sort_idx]

    # Plot for each parameter
    for pname in param_names:
        means = np.array(param_means[pname])[sort_idx]
        stds = np.array(param_stds[pname])[sort_idx]
        plt.figure()
        plt.errorbar(m1_arr, means, yerr=stds, fmt='o-', capsize=3)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("m1")
        plt.ylabel(f"Precision ({pname})")
        plt.title(f"Precision of {pname} vs m1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"precision_vs_m1_{pname}.png",dpi=300)
        plt.close()