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
        source_f = "snr_0/"
    
    print("SNR dictionary file not found. Processing SNR data...")
    snr_dict = {}
    print(f"Processing source folder: {source_f}")
    folders = glob.glob(f"{source_f}/m1*/")
    print(f"Found {len(folders)} files for source.")
    for i,fold in enumerate(folders):
        print(f"Processing source, file {i+1}/{len(folders)}: {fold}")
        
        if i == 0:
            snr_dict['snr'] = []
            snr_dict['redshift'] = []
            snr_dict['dist'] = []
            snr_dict['detector_params'] = []
            snr_dict['source_params'] = []
            snr_dict['e_f'] = h5py.File(sorted(glob.glob(f"{fold}/*/snr.h5"))[0], "r")["e_f"][()]
            snr_dict['Tpl'] = h5py.File(sorted(glob.glob(f"{fold}/*/snr.h5"))[0], "r")["T"][()]
        
        snr_list = np.asarray([h5py.File(el, "r")["snr"][()] for el in sorted(glob.glob(f"{fold}/*/snr.h5"))])
        redshift = h5py.File(sorted(glob.glob(f"{fold}/*/snr.h5"))[0], "r")["redshift"][()]
        # params
        detector_params = np.asarray([h5py.File(el, "r")["parameters"][()] for el in sorted(glob.glob(f"{fold}/*/snr.h5"))])
        source_params = detector_params.copy()
        source_params[:,0] = source_params[:,0] / (1 + redshift)
        source_params[:,1] = source_params[:,1] / (1 + redshift)
        dist = detector_params[:,6]
        # save to dict
        snr_dict['snr'].append(snr_list)
        snr_dict['redshift'].append(redshift)
        snr_dict['dist'].append(dist)
        snr_dict['detector_params'].append(detector_params)
        snr_dict['source_params'].append(source_params)
    
    ind_sort = np.argsort(snr_dict['redshift'])
    print("Length of redshift array before sorting:", len(snr_dict['redshift']))
    for key, item in snr_dict.items():
        if key != 'e_f' and key != 'Tpl':
            snr_dict[key] = np.asarray(snr_dict[key])[ind_sort]
    
    print("Finished processing SNR data.")
    snr_dict['m1'] = snr_dict['source_params'][0,0,0]
    snr_dict['m2'] = snr_dict['source_params'][0,0,1]
    # add the sign to the spin
    snr_dict['a'] = snr_dict['source_params'][0,0,2] * snr_dict['source_params'][0,0,5]
    snr_dict['p0'] = snr_dict['source_params'][0,0,3]
    snr_dict['e0'] = snr_dict['source_params'][0,0,4]
    snr_dict['dist'] = snr_dict['dist'][:,0]
    snr = snr_dict['snr']
    redshift = snr_dict['redshift']
    quantile = 0.68
    snr_dict['quantile'] = quantile
    snr_ref_value = 30.0
    snr_dict['snr_ref_value'] = snr_ref_value
    z_ref_value = 0.5
    snr_dict['z_ref_value'] = z_ref_value
    
    snr_median = np.median(snr, axis=-1)
    snr_m_sigma = np.quantile(snr, (1-quantile)/2, axis=-1)
    snr_p_sigma = np.quantile(snr, 1-(1-quantile)/2, axis=-1)
    snr_dict['snr_median'] = snr_median
    snr_dict['snr_m_sigma'] = snr_m_sigma
    snr_dict['snr_p_sigma'] = snr_p_sigma
    
    z_ref_median = np.exp(np.interp(np.log(snr_ref_value), np.log(snr_median[np.argsort(snr_median)]), np.log(redshift[np.argsort(snr_median)]), left=np.nan, right=np.nan))
    z_ref_p_sigma = np.exp(np.interp(np.log(snr_ref_value), np.log(snr_p_sigma[np.argsort(snr_p_sigma)]), np.log(redshift[np.argsort(snr_p_sigma)]), left=np.nan, right=np.nan))
    z_ref_m_sigma = np.exp(np.interp(np.log(snr_ref_value), np.log(snr_m_sigma[np.argsort(snr_m_sigma)]), np.log(redshift[np.argsort(snr_m_sigma)]), left=np.nan, right=np.nan))
    print(f"At SNR={snr_ref_value}, inferred redshift median: {z_ref_median}, +sigma: {z_ref_p_sigma}, -sigma: {z_ref_m_sigma}")
    snr_dict['redshift_ref_median'] = z_ref_median
    snr_dict['redshift_ref_p_sigma'] = z_ref_p_sigma
    snr_dict['redshift_ref_m_sigma'] = z_ref_m_sigma
    
    snr_at_z_ref_median = np.exp(np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_median[np.argsort(redshift)]), left=np.nan, right=np.nan))
    snr_at_z_ref_p_sigma = np.exp(np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_p_sigma[np.argsort(redshift)]), left=np.nan, right=np.nan))
    snr_at_z_ref_m_sigma = np.exp(np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_m_sigma[np.argsort(redshift)]), left=np.nan, right=np.nan))
    print(f"At redshift={z_ref_value}, inferred SNR median: {snr_at_z_ref_median}, +sigma: {snr_at_z_ref_p_sigma}, -sigma: {snr_at_z_ref_m_sigma}")
    snr_dict['snr_at_z_ref_median'] = snr_at_z_ref_median
    snr_dict['snr_at_z_ref_p_sigma'] = snr_at_z_ref_p_sigma
    snr_dict['snr_at_z_ref_m_sigma'] = snr_at_z_ref_m_sigma
    
    # Save snr_dict to HDF5 file
    with h5py.File(source_f + "/detection.h5", "w") as f:
        for key, value in snr_dict.items():
            f.create_dataset(key, data=value)
    print(f"SNR data saved to {source_f}/detection.h5")
    
    # open    
    with open("so3_sources_Dec8.json", "r") as f:
        source_dict = json.load(f)
    
    out_dict = {}
    update_key = source_f.strip("/").split("_")[-1]
    for key in source_dict.keys():
        if key == update_key:
            out_dict[key] = source_dict[key].copy()
            out_z = snr_dict['redshift_ref_median']
            if np.isnan(out_z):
                out_z = -1.0
            out_dict[key]['z_ref_median'] = out_z
    
    with open(source_f + "inference_so3_sources_Dec8.json", "w") as f:
        json.dump(out_dict, f, indent=4)
