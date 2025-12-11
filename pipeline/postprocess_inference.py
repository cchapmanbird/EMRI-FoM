import h5py
import numpy as np
import glob
import json
import os
import sys
from scipy.interpolate import interp1d

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    plt.style.use('science')

    # Get source_f from command line argument
    if len(sys.argv) > 1:
        source_f = sys.argv[1]
    else:
        source_f = "inference_0/"
    
    print("SNR dictionary file not found. Processing SNR data...")
    inference_dict = {}
    print(f"Processing source folder: {source_f}")
    folders = glob.glob(f"{source_f}/m1*/")
    print(f"Found {len(folders)} files for source.")
    for i,fold in enumerate(folders):
        print(f"Processing source, file {i+1}/{len(folders)}: {fold}")
        len_realizations = len(sorted(glob.glob(f"{fold}/*/results.npz")))
        if len_realizations != 1000:
            print(f"Incomplete data in {fold}, found {len_realizations} realizations. Skipping.")
            continue  # skip incomplete data
        e_f = float(np.load(sorted(glob.glob(f"{fold}/*/snr.npz"))[0])["e_f"])
        param_names = np.array(np.load(sorted(glob.glob(f"{fold}/*/results.npz"))[0])["names"],dtype='str')
        
        snr_list = np.asarray([h5py.File(el, "r")["snr"][()] for el in sorted(glob.glob(f"{fold}/*/snr.h5"))])
        redshift = h5py.File(sorted(glob.glob(f"{fold}/*/snr.h5"))[0], "r")["redshift"][()]
        # extract results
        source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{fold}/*/results.npz"))])
        detector_cov = np.asarray([np.load(el)["cov"] for el in sorted(glob.glob(f"{fold}/*/results.npz"))])
        fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{fold}/*/results.npz"))])
        fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
        fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
        source_measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
        detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
        
        # identify parameter names and adjust for sky location and inclination
        names = param_names.tolist()
        assert source_measurement_precision.shape[-1] == len(names)
        assert detector_measurement_precision.shape[-1] == len(names)

        ind_sky = [names.index('qS'), names.index('phiS')]
        ind_volume = [names.index('dist'), names.index('qS'), names.index('phiS')]
        
        # skylocation error estimation
        qS = fish_params[:, ind_sky[0]]
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
        
        if e_f == 0.0:
            key = 'circular'
        else:
            key = 'eccentric'
            
        detector_params = np.asarray([h5py.File(el, "r")["parameters"][()] for el in sorted(glob.glob(f"{fold}/*/snr.h5"))])
        Tpl = h5py.File(sorted(glob.glob(f"{fold}/*/snr.h5"))[0], "r")["T"][()]
        
        inference_dict[key] = {}
        inference_dict[key]['param_names'] = names
        inference_dict[key]['source_measurement_precision'] = source_measurement_precision
        inference_dict[key]['detector_measurement_precision'] = detector_measurement_precision
        inference_dict[key]['fish_params'] = fish_params
        
        inference_dict[key]['e_f'] = e_f
        inference_dict[key]['snr'] = snr_list
        inference_dict[key]['redshift'] = redshift
    
        # main parameters
        inference_dict[key]['m1'] = detector_params[0,0]/(1+redshift)
        inference_dict[key]['m2'] = detector_params[0,1]/(1+redshift)
        # add the sign to the spin
        inference_dict[key]['a'] = detector_params[0,2] * detector_params[0,5]
        inference_dict[key]['p0'] = detector_params[0,3]
        inference_dict[key]['e0'] = detector_params[0,4]
        inference_dict[key]['Tpl'] = Tpl
        inference_dict[key]['dist'] = detector_params[0,6]
        inference_dict[key]['redshift'] = redshift
    
    # Save inference_dict to HDF5 file
    with h5py.File(source_f + "/inference.h5", "w") as f:
        for key, value in inference_dict.items():
            print(f"Saving key: {key}")
            if key in ['circular', 'eccentric']:
                grp = f.create_group(key)
                for subkey, subvalue in value.items():
                    grp.create_dataset(subkey, data=subvalue)
            else:
                f.create_dataset(key, data=value)
    print(f"SNR data saved to {source_f}/inference.h5")