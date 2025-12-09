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
import time 
full_names = np.array(['m1','m2','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0', 'A', 'nr'])
list_folders = sorted(glob.glob("./snr*/*/")) # sorted(glob.glob("./inference*/*/")) + 
# print(list_folders)
list_results = []
h5_path = "so3_snr_sources_Dec8.h5"
# check if the HDF5 file already exists
if os.path.exists(h5_path):
    print(f"HDF5 file {h5_path} already exists.")
else:
    with h5py.File(h5_path, "w") as h5f:
        # Assessment Process
        for source in list_folders:
            if os.path.isdir(source) is False:
                continue
            print(f"Processing {source}")
            tic = time.time()
            # Tpl = float(source.split("T=")[-1].split("_")[0])
            # redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
            # detector_params = np.asarray([np.load(el)["parameters"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
            # e_f = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["e_f"]
            redshift = h5py.File(sorted(glob.glob(f"{source}/*/snr.h5"))[0], "r")["redshift"][()]
            detector_params = np.asarray([h5py.File(el, "r")["parameters"][()] for el in sorted(glob.glob(f"{source}/*/snr.h5"))])
            e_f = h5py.File(sorted(glob.glob(f"{source}/*/snr.h5"))[0], "r")["e_f"][()]
            Tpl = h5py.File(sorted(glob.glob(f"{source}/*/snr.h5"))[0], "r")["T"][()]
            
            source_params = detector_params[0].copy()
            source_params[0] = source_params[0] / (1 + redshift)
            source_params[1] = source_params[1] / (1 + redshift)
            lum_dist = detector_params[:,6]
            sky_loc = detector_params[:,7:9]
            spin_loc = detector_params[:, 9:11]
            detector_params = detector_params[0]
            toc = time.time()
            print(f"Loaded source parameters in {toc - tic:.2f} seconds.")
            tic = time.time()
            # Load SNRs
            snr_list = np.asarray([h5py.File(el, "r")["snr"][()] for el in sorted(glob.glob(f"{source}/*/snr.h5"))])
            # snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
            toc = time.time()
            print(f"Loaded SNRs in {toc - tic:.2f} seconds.")
            tic = time.time()
            # Prepare result for snr
            result = {}
            result["source_params"] = source_params
            result["detector_params"] = detector_params
            result["full_names"] = full_names
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
            toc = time.time()
            print(f"Stored results in {toc - tic:.2f} seconds.")
            # tic = time.time()
            # # SNR plot
            # plt.figure()
            # plt.hist(snr_list, bins=30)
            # plt.xlabel('SNR')
            # plt.ylabel('Counts')
            # plt.savefig(f"{source}/snr_histogram.png",dpi=300)
            # plt.figure()
            # toc = time.time()
            # print(f"Plotted SNR histogram in {toc - tic:.2f} seconds.")
            # plt.close('all')
            
            if "inference" in source:
                # Fisher matrices and covariances
                param_names = np.asarray([np.load(el)["names"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                detector_cov = np.asarray([np.load(el)["cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
                fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
                fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
                source_measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
                detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
                # identify parameter names and adjust for sky location and inclination
                names = param_names[0].tolist()
                ind_sky = [names.index('qS'), names.index('phiS')]
                ind_volume = [names.index('dist'), names.index('qS'), names.index('phiS')]
                
                # skylocation error estimation
                qS = fish_params[:, ind_sky[0]]
                assert np.all(result["sky_loc"][:,0]==fish_params[:, ind_sky[0]])
                Sigma = source_cov[:,ind_sky[0]:ind_sky[1]+1, ind_sky[0]:ind_sky[1]+1]
                err_sky_loc = 2 * np.pi * np.sin(qS) * np.sqrt(np.linalg.det(Sigma)) * (180.0 / np.pi) ** 2
                names.append("OmegaS")
                source_measurement_precision = np.hstack((source_measurement_precision, err_sky_loc[:, None]))
                detector_measurement_precision = np.hstack((detector_measurement_precision, err_sky_loc[:, None]))
                
                # volume error estimation
                Sigma_V = source_cov[:,ind_volume[0]:ind_volume[2]+1, ind_volume[0]:ind_volume[2]+1]
                err_volume = (4/3) * np.pi * (fish_params[:,ind_volume[0]])**2 * np.sqrt(np.linalg.det(Sigma_V))
                names.append("DeltaV")
                source_measurement_precision = np.hstack((source_measurement_precision, err_volume[:, None]))
                detector_measurement_precision = np.hstack((detector_measurement_precision, err_volume[:, None]))
                
                # print(source_measurement_precision.shape[-1], len(names), detector_measurement_precision.shape[-1])
                assert source_measurement_precision.shape[-1] == len(names)
                assert detector_measurement_precision.shape[-1] == len(names)

                # Save errors in HDF5
                grp.create_dataset("error_source", data=source_measurement_precision)
                grp.create_dataset("error_detector", data=detector_measurement_precision)
                grp.create_dataset("error_names", data=names)
                plt.close('all')
    print(f"Results saved in {h5_path}")
