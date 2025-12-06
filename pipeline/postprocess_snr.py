import h5py
import numpy as np
import glob
import json
import os
from scipy.interpolate import interp1d

def find_matching_sources(input_params, source_dict):
    """
    Finds all source numbers that match the given input parameters.
    
    :param input_params: dict of parameters to match (e.g., {"m1": 50000.0, "a": 0.99})
    :param source_dict: dict of sources
    :return: list of matching source numbers
    """
    matches = []
    for source_num, source_params in source_dict.items():
        # Check if all keys in input_params match the source_params
        if all(abs(source_params[key] - input_params[key]) < 1e-9 for key in input_params):
            matches.append(source_num)
    return np.asarray(matches,dtype=int)


def get_results(snr_dict, input_dict, quantile = 0.68, snr_ref_value = 30.0, z_ref_value = 0.001):
    """Get results for matching sources from snr_dict.
    Args:
        snr_dict (dict): Dictionary containing SNR data for sources.
        input_dict (dict): Dictionary of input parameters to match sources.
        quantile (float): Quantile for error bars.
        snr_ref_value (float): Reference SNR value at which to compute the reference z and DL
        z_ref_value (float): Reference redshift value at which to compute the reference SNR
    Returns:
        dict: Dictionary containing results for matching sources.
    """
    ind_s = find_matching_sources(input_dict, snr_dict)
    m1 = np.asarray([snr_dict[source_n]['m1'] for source_n in ind_s])
    m2 = np.asarray([snr_dict[source_n]['m2'] for source_n in ind_s])
    a =  np.asarray([snr_dict[source_n]['a'] for source_n in ind_s])
    Tpl = np.asarray([snr_dict[source_n]['Tpl'] for source_n in ind_s])
    e0 = np.asarray([snr_dict[source_n]['e0'] for source_n in ind_s])
    ef = np.asarray([snr_dict[source_n]['e_f'] for source_n in ind_s])
    snr_shape = np.asarray([snr_dict[source_n]['snr'].shape for source_n in ind_s])
    if np.all(snr_shape == (10, 100)) == False:
        raise ValueError("SNR shape mismatch. Expected (10, 100) for all sources.")
    snr = np.asarray([snr_dict[source_n]['snr'] for source_n in ind_s])
    if np.isnan(snr).sum() > 0:
        print("Warning: NaN values found in SNR data.")
    snr_median = np.median(snr, axis=-1)
    snr_m_sigma = np.quantile(snr, (1-quantile)/2, axis=-1)
    snr_p_sigma = np.quantile(snr, 1-(1-quantile)/2, axis=-1)
    redshift = snr_dict[0]['redshift']
    dl = snr_dict[0]['DL']
    sky_loc =  snr_dict[0]['sky_loc']
    spin_loc =  snr_dict[0]['spin_loc']
    
    if snr_ref_value < snr_median.min() or snr_ref_value > snr_median.max():
        print("Warning: z_ref_value is out of bounds of the redshift array.")
        print(f"snr_ref_value {snr_ref_value} is out of bounds ({snr_median.min()}, {snr_median.max()})")
    
    z_ref_median = np.exp(np.asarray([np.interp(np.log(snr_ref_value), np.log(snr_[np.argsort(snr_)]), np.log(redshift[np.argsort(snr_)]), left=np.nan, right=np.nan) for snr_ in snr_median]))
    z_ref_p_sigma = np.exp(np.asarray([np.interp(np.log(snr_ref_value), np.log(snr_[np.argsort(snr_)]), np.log(redshift[np.argsort(snr_)]), left=np.nan, right=np.nan) for snr_ in snr_p_sigma]))
    z_ref_m_sigma = np.exp(np.asarray([np.interp(np.log(snr_ref_value), np.log(snr_[np.argsort(snr_)]), np.log(redshift[np.argsort(snr_)]), left=np.nan, right=np.nan) for snr_ in snr_m_sigma]))
    
    if z_ref_value < redshift.min() or z_ref_value > redshift.max():
        print("Warning: z_ref_value is out of bounds of the redshift array.")
        print(f"z_ref_value {z_ref_value} is out of bounds ({redshift.min()}, {redshift.max()})")
    
    snr_at_z_ref_median = np.exp(np.asarray([np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_[np.argsort(redshift)]), left=np.nan, right=np.nan) for snr_ in snr_median]))
    snr_at_z_ref_p_sigma = np.exp(np.asarray([np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_[np.argsort(redshift)]), left=np.nan, right=np.nan) for snr_ in snr_p_sigma]))
    snr_at_z_ref_m_sigma = np.exp(np.asarray([np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_[np.argsort(redshift)]), left=np.nan, right=np.nan) for snr_ in snr_m_sigma]))
    
    # lum_d = cosmo.get_luminosity_distance(z)

    return_dict = {"m1": m1, "m2": m2, "a": a, "Tpl": Tpl, "e0": e0 , "e_f": ef, 
                   "snr": snr,
                   "redshift_ref_median": z_ref_median, "redshift_ref_m_sigma": z_ref_m_sigma, "redshift_ref_p_sigma": z_ref_p_sigma,
                   "snr_at_z_ref_median": snr_at_z_ref_median, "snr_at_z_ref_m_sigma": snr_at_z_ref_m_sigma, "snr_at_z_ref_p_sigma": snr_at_z_ref_p_sigma,
                   "snr_median": snr_median, "snr_m_sigma": snr_m_sigma, "snr_p_sigma": snr_p_sigma, 
                   "redshift": redshift, "dl": dl, "spin_loc": spin_loc, "sky_loc": sky_loc,
                  "ind_s": ind_s
                  }
    return return_dict

def load_snr_dict_from_h5(filename="snr_sources.h5"):
    """Load snr_dict from HDF5 file.
    
    Args:
        filename (str): Path to the HDF5 file
        
    Returns:
        dict: The reconstructed snr_dict
    """
    snr_dict = {}
    with h5py.File(filename, "r") as f:
        for source_key in f.keys():
            source_n = int(source_key.split("_")[1])
            snr_dict[source_n] = {}
            grp = f[source_key]
            for key in grp.keys():
                snr_dict[source_n][key] = grp[key][...]
    return snr_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    plt.style.use('science')


    if os.path.exists("snr_sources.h5"):
        print("Loading existing SNR dictionary from file...")
        snr_dict_loaded = load_snr_dict_from_h5("snr_sources.h5")
        print("SNR dictionary loaded successfully.")
    else:
        print("SNR dictionary file not found. Processing SNR data...")
        
        snr_dict = {}
        source_folders = sorted([d for d in glob.glob(f"production_snr_*") if os.path.isdir(d)])
        for source_f in source_folders:
            print(f"Processing source folder: {source_f}")
            source_n = source_f.split('_')[2]
            folders = glob.glob(f"{source_f}/m1*/aggregated_results.h5")
            print(f"Found {len(folders)} files for source {source_n}.")
            for i,fold in enumerate(folders):
                print(f"Processing source {source_n}, file {i+1}/{len(folders)}: {fold}")
                results = h5py.File(fold, "r")['SNR_analysis']
                if i == 0:
                    snr_dict[source_n] = {}
                    snr_dict[source_n]['snr'] = []
                    snr_dict[source_n]['redshift'] = []
                    snr_dict[source_n]['DL'] = []
                    for key, item in results.items():
                        if (key != 'snr') and (key != 'redshift') and (key != 'DL'):
                            snr_dict[source_n][key] = item[...]
                    # print(snr_dict[source_n]['sky_loc'][0])

                snr_dict[source_n]['snr'].append(results['snr'][...])
                snr_dict[source_n]['redshift'].append(results['redshift'][...])
                snr_dict[source_n]['DL'].append(results['DL'][...])
            
            ind_sort = np.argsort(snr_dict[source_n]['redshift'])
            print("Length of redshift array before sorting:", len(snr_dict[source_n]['redshift']))
            for key, item in snr_dict[source_n].items():
                if (key != 'snr') and (key != 'redshift') and (key != 'DL'):
                    continue
                snr_dict[source_n][key] = np.asarray(snr_dict[source_n][key])[ind_sort]
        
        print("Finished processing SNR data for all sources.")
    
        # Save snr_dict to HDF5 file
        with h5py.File("snr_sources.h5", "w") as f:
            for source_n, source_data in snr_dict.items():
                grp = f.create_group(f"source_{source_n}")
                for key, value in source_data.items():
                    grp.create_dataset(key, data=value)
        
        print("SNR dictionary saved to snr_sources.h5")
        snr_dict_loaded = load_snr_dict_from_h5("snr_sources.h5")
        print("SNR dictionary loaded from snr_sources.h5")
    
    
    input_params = {}
    matching_sources = find_matching_sources(input_params, snr_dict_loaded)
    print("Matching source numbers:", len(matching_sources))
    
    # open    
    with open("so3_snr_sources.json", "r") as f:
        source_dict = json.load(f)
    
    unique_m1 = sorted(set([source_dict[i]['m1'] for i in source_dict]))
    unique_m2 = sorted(set([source_dict[i]['m2'] for i in source_dict]))
    unique_a = sorted(set([source_dict[i]['a'] for i in source_dict]))
    unique_e0 = sorted(set([source_dict[i]['e_0'] for i in source_dict]))
    unique_Tpl = sorted(set([source_dict[i]['Tpl'] for i in source_dict]))
    
    # # save z_ref_median in source_dict
    temp_dict = get_results(snr_dict_loaded, {})
    
    out_dict = source_dict.copy()
    number_included = 0
    for key in source_dict.keys():
        # print("Processing source", key)
        mask = np.prod(np.asarray([(np.abs(1-temp_dict[load_key] / source_dict[key][load_key])<1e-6) for load_key in ['m1', 'm2', 'a', 'Tpl']]),axis=0, dtype=bool)
        if len(temp_dict['redshift_ref_median'][mask]) != 1:
            print("No matching source found for source", key)
            breakpoint()
        if np.isnan(temp_dict['redshift_ref_median'][mask][0]) == False:
            print("Mask", len(temp_dict['redshift_ref_median'][mask]), temp_dict['redshift_ref_median'][mask])
            out_dict[key]['z_ref_median'] = temp_dict['redshift_ref_median'][mask][0]
            number_included += 1
        else:
            print("No valid z_ref_median for source", key, "removing from output.")
            del out_dict[key]
    print(f"Included {number_included} sources with valid z_ref_median out of {len(source_dict)} total sources.")
    with open("so3_inference_sources_with_z_ref.json", "w") as f:
        json.dump(out_dict, f, indent=4)
