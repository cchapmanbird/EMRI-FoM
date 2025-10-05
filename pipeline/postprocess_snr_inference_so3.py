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


list_folders = sorted(glob.glob("./production_inference_*e_f=*")) + sorted(glob.glob("./production_snr_m*"))
# print(list_folders)
list_results = []
h5_path = "so3_results.h5"
# check if the HDF5 file already exists
if os.path.exists(h5_path):
    print(f"HDF5 file {h5_path} already exists.")
else:
    with h5py.File(h5_path, "w") as h5f:
        # Assessment Process
        for source in list_folders:
            print(f"Processing {source}...")
            Tpl = float(source.split("T=")[-1].split("_")[0])
            redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
            detector_params = np.asarray([np.load(el)["parameters"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
            e_f = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["e_f"]
            source_params = detector_params[0].copy()
            source_params[0] = source_params[0] / (1 + redshift)
            source_params[1] = source_params[1] / (1 + redshift)
            lum_dist = detector_params[:,6]
            sky_loc = detector_params[:,7:9]
            spin_loc = detector_params[:, 9:11]
            detector_params = detector_params[0]
            snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
            
            # Prepare result for snr
            result = {
                "m1": source_params[0],
                "m2": source_params[1],
                "a": source_params[2]*source_params[5],
                "p0": source_params[3],
                "e0": source_params[4],
                "DL": source_params[6],
                "e_f": e_f,
                "Tpl": Tpl,
                "redshift": redshift,
                "lum_dist": lum_dist,
                "snr": snr_list,
                "sky_loc": sky_loc,
                "spin_loc": spin_loc,
            }

            # Store in HDF5 the main values
            grp = h5f.create_group(source)
            for k, v in result.items():
                grp.create_dataset(k, data=v)


            # SNR plot
            plt.figure()
            plt.hist(snr_list, bins=30)
            plt.xlabel('SNR')
            plt.ylabel('Counts')
            plt.savefig(f"{source}/snr_histogram.png",dpi=300)
            plt.figure()

            if "inference" in source:
                # Fisher matrices and covariances
                source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                detector_cov = np.asarray([np.load(el)["cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
                fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
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


                for ii, el in enumerate(names):
                    error_source = measurement_precision[:, ii]
                    error_detector = detector_measurement_precision[:, ii]
                    if (el == "M") or (el == "mu"):
                        error_source = error_source / source_params[ii]
                        error_detector = error_detector / detector_params[ii]
                        xlabel = 'Relative error ' + el
                        group_name = f"relative_errors_{el}"
                    else:
                        xlabel = 'Absolute error ' + el
                        group_name = f"absolute_errors_{el}"

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

                    # # plot of SNR across the sky
                    # plt.figure()
                    # nside = 8
                    # npix = hp.nside2npix(nside)
                    # sky_map = np.zeros(npix)
                    # theta = sky_loc[:, 0]
                    # phi = sky_loc[:, 1]
                    # pixels = hp.ang2pix(nside, theta, phi)
                    # # Accumulate the sum of errors in each pixel
                    # for i, pix in enumerate(pixels):
                    #     sky_map[pix] += error_source[i]
                    # counts = np.bincount(pixels, minlength=npix)
                    # counts[counts == 0] = 1  # avoid division by zero
                    # sky_map = sky_map / counts  # average error per pixel
                    # hp.mollview(sky_map, title=f"Precision error across sky for {el}", unit="Error", cmap="viridis")
                    # plt.savefig(f"{source}/precision_error_sky_{el}.png",dpi=300)
                    # plt.close()

                    # Save distributions
                    np.savez(f"{source}/{el}_distribution.npz", error_source=error_source, error_detector=error_detector)
                    # Save errors in HDF5
                    err_grp = grp.create_group(group_name)
                    err_grp.create_dataset("error_source", data=error_source)
                    err_grp.create_dataset("error_detector", data=error_detector)
                    plt.close('all')
    print(f"Results saved in {h5_path}")
