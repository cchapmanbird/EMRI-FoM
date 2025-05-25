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
    plt.legend()
    plt.savefig(f"{source}/snr_histogram.png")
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
    plt.savefig(f"{source}/snr_sky.png")
    plt.close()

    
    measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
    detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
    err_sky_loc = np.asarray([np.load(el)["err_sky_loc"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
    names = np.asarray([np.load(el)["names"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])[0]
    names[6] = "Omega"
    # define up to sky location
    names = names[:7]
    measurement_precision[:, 6] = err_sky_loc
    detector_measurement_precision[:, 6] = err_sky_loc  # Assuming sky loc error is same for both

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
        plt.savefig(f"{source}/{el}_histogram.png")
        plt.close()

        # SNR vs error plot
        plt.figure()
        plt.plot(snr_list, error_source, 'o', label='Source frame')
        plt.plot(snr_list, error_detector, 'x', label='Detector frame')
        snr_vec = np.linspace(np.min(snr_list), np.max(snr_list), 100)
        if el == "dist":
            plt.plot(snr_vec, lum_dist / snr_vec, 'r--', label='d/SNR')
        else:
            plt.plot(snr_vec, 1 / snr_vec, 'r--', label='1/SNR')
        plt.xlabel('SNR')
        plt.ylabel(xlabel)
        plt.legend()
        plt.savefig(f"{source}/snr_{el}.png")
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
        plt.savefig(f"{source}/precision_error_sky_{el}.png")
        plt.close()

        # Save distributions
        np.savez(f"{source}/{el}_distribution.npz", error_source=error_source, error_detector=error_detector)

