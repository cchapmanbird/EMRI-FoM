import h5py
import numpy as np
import glob
import json
import os
from scipy.interpolate import interp1d

with open("emri_pe_sources.json", "r") as f:
    source_dict = json.load(f)


def find_matching_sources(input_params):
    """
    Finds all source numbers that match the given input parameters.
    
    :param input_params: dict of parameters to match (e.g., {"m1": 50000.0, "a": 0.99})
    :param snr_dict: dict of sources
    :return: list of matching source numbers
    """
    matches = []
    for source_num, source_params in source_dict.items():
        # Check if all keys in input_params match the source_params
        if all(abs(source_params[key] - input_params[key]) < 1e-9 for key in input_params):
            matches.append(source_num)
    return np.asarray(matches,dtype=int)


def get_results(snr_dict, input_dict, quantile = 0.68, snr_ref_value = 30.0, z_ref_value = 1.0):
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
    
    ind_s = find_matching_sources(input_dict)
    m1 = np.asarray([snr_dict[source_n]['m1'] for source_n in ind_s])
    m2 = np.asarray([snr_dict[source_n]['m2'] for source_n in ind_s])
    a =  np.asarray([snr_dict[source_n]['a'] for source_n in ind_s])
    Tpl = np.asarray([snr_dict[source_n]['Tpl'] for source_n in ind_s])
    e0 = np.asarray([snr_dict[source_n]['e0'] for source_n in ind_s])
    ef = np.asarray([snr_dict[source_n]['e_f'] for source_n in ind_s])
    snr = np.asarray([snr_dict[source_n]['snr'] for source_n in ind_s])
    snr_median = np.median(snr, axis=-1)
    snr_m_sigma = np.quantile(snr, (1-quantile)/2, axis=-1)
    snr_p_sigma = np.quantile(snr, 1-(1-quantile)/2, axis=-1)
    redshift = snr_dict[0]['redshift']
    dl = snr_dict[0]['DL']
    sky_loc =  snr_dict[0]['sky_loc']
    spin_loc =  snr_dict[0]['spin_loc']
    
    z_ref_median = np.exp(np.asarray([np.interp(np.log(snr_ref_value), np.log(snr_[np.argsort(snr_)]), np.log(redshift[np.argsort(snr_)]), left=np.nan, right=np.nan) for snr_ in snr_median]))
    z_ref_p_sigma = np.exp(np.asarray([np.interp(np.log(snr_ref_value), np.log(snr_[np.argsort(snr_)]), np.log(redshift[np.argsort(snr_)]), left=np.nan, right=np.nan) for snr_ in snr_p_sigma]))
    z_ref_m_sigma = np.exp(np.asarray([np.interp(np.log(snr_ref_value), np.log(snr_[np.argsort(snr_)]), np.log(redshift[np.argsort(snr_)]), left=np.nan, right=np.nan) for snr_ in snr_m_sigma]))
    
    snr_at_z_ref_median = np.exp(np.asarray([np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_[np.argsort(redshift)]), left=np.nan, right=np.nan) for snr_ in snr_median]))
    snr_at_z_ref_p_sigma = np.exp(np.asarray([np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_[np.argsort(redshift)]), left=np.nan, right=np.nan) for snr_ in snr_p_sigma]))
    snr_at_z_ref_m_sigma = np.exp(np.asarray([np.interp(np.log(z_ref_value), np.log(redshift[np.argsort(redshift)]), np.log(snr_[np.argsort(redshift)]), left=np.nan, right=np.nan) for snr_ in snr_m_sigma]))
    
    # lum_d = cosmo.get_luminosity_distance(z)

    return_dict = {"m1": m1, "m2": m2, "a": a, "Tpl": Tpl, "e_0": e0 , "e_f": ef, 
                   "snr": snr,
                   "redshift_ref_median": z_ref_median, "redshift_ref_m_sigma": z_ref_m_sigma, "redshift_ref_p_sigma": z_ref_p_sigma,
                   "snr_at_z_ref_median": snr_at_z_ref_median, "snr_at_z_ref_m_sigma": snr_at_z_ref_m_sigma, "snr_at_z_ref_p_sigma": snr_at_z_ref_p_sigma,
                   "snr_median": snr_median, "snr_m_sigma": snr_m_sigma, "snr_p_sigma": snr_p_sigma, 
                   "redshift": redshift, "dl": dl, "spin_loc": spin_loc, "sky_loc": sky_loc,
                  "ind_s": ind_s
                  }
    return return_dict

def load_snr_dict_from_h5(filename="snr_dict_emri_pe_sources.h5"):
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
    input_params = {}
    matching_sources = find_matching_sources(input_params)
    print("Matching source numbers:", len(matching_sources))

    if os.path.exists("snr_dict_emri_pe_sources.h5"):
        print("Loading existing SNR dictionary from file...")
        snr_dict_loaded = load_snr_dict_from_h5("snr_dict_emri_pe_sources.h5")
        print("SNR dictionary loaded successfully.")
    else:
        print("SNR dictionary file not found. Processing SNR data...")
        
        snr_dict = {source_n: {} for source_n in range(0, 100)}
        source_n = 0
        for source_n in range(0, 100):
            folders = glob.glob(f"production_snr_{source_n}/m1*/*.h5")
            for i,fold in enumerate(folders):
                results = h5py.File(fold, "r")['SNR_analysis']
                if i == 0:
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
            for key, item in snr_dict[source_n].items():
                if (key != 'snr') and (key != 'redshift') and (key != 'DL'):
                    continue
                snr_dict[source_n][key] = np.asarray(snr_dict[source_n][key])[ind_sort]
        
        print("Finished processing SNR data for all sources.")
    
        # Save snr_dict to HDF5 file
        with h5py.File("snr_dict_emri_pe_sources.h5", "w") as f:
            for source_n, source_data in snr_dict.items():
                grp = f.create_group(f"source_{source_n}")
                for key, value in source_data.items():
                    grp.create_dataset(key, data=value)
        
        print("SNR dictionary saved to snr_dict_emri_pe_sources.h5")
        snr_dict_loaded = load_snr_dict_from_h5("snr_dict_emri_pe_sources.h5")
        print("SNR dictionary loaded from snr_dict_emri_pe_sources.h5")
    
    dict_out = get_results(snr_dict_loaded, {})
    for i, source_n in enumerate(dict_out["ind_s"]):
        if np.isnan(dict_out["redshift_ref_median"][i]):
            source_dict[str(source_n)]["z_ref_median"] = -1.0
        else:
            source_dict[str(source_n)]["z_ref_median"] = dict_out["redshift_ref_median"][i]
    # save updated source_dict with z_ref_median
    with open("emri_pe_sources_with_z_ref.json", "w") as f:
        json.dump(source_dict, f, indent=4)
    
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use('science')

    input_dict = {"Tpl": 0.25, "e_0": 0.0}
    plt.figure()
    plt.loglog(get_results(snr_dict_loaded, input_dict)["m1"],get_results(snr_dict_loaded, input_dict)["m2"], 'o')
    plt.xlabel("m1")
    plt.ylabel("m2")
    plt.savefig("test_plot.png")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    list_m2 = [1, 5, 10, 50, 100, 1000, 1e4]
    list_marker = ["o", "v", "P", "X", "*", "^", 'D']
    
    # Plot for a = 0.99
    for m2_, fmt in zip(list_m2, list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": 0.99, "m2": m2_}
        dict_out = get_results(snr_dict_loaded, input_dict)
        ax1.errorbar(dict_out["m1"], dict_out["snr_at_z_ref_median"], 
                     yerr=[dict_out["snr_at_z_ref_median"] - dict_out["snr_at_z_ref_m_sigma"], 
                           dict_out["snr_at_z_ref_p_sigma"] - dict_out["snr_at_z_ref_median"]], 
                     linestyle='none', capsize=7, fmt=fmt, label=f'm2={m2_}')
    ax1.set_ylabel("SNR")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()
    ax1.set_title("a = 0.99")
    ax1.legend()
    
    # Plot for a = -0.99
    for m2_, fmt in zip(list_m2, list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": -0.99, "m2": m2_}
        dict_out = get_results(snr_dict_loaded, input_dict)
        ax2.errorbar(dict_out["m1"], dict_out["snr_at_z_ref_median"], 
                     yerr=[dict_out["snr_at_z_ref_median"] - dict_out["snr_at_z_ref_m_sigma"], 
                           dict_out["snr_at_z_ref_p_sigma"] - dict_out["snr_at_z_ref_median"]], 
                     linestyle='none', capsize=7, fmt=fmt, label=f'm2={m2_}')
    ax2.set_ylabel("SNR")
    ax2.set_xlabel("m1")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid()
    ax2.set_title("a = -0.99")
    ax2.legend()
    ax2.set_xlim(2e4, 2e7)
    plt.tight_layout()
    plt.savefig("snr_vs_m1.png")
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    list_m2 = [1, 5, 10, 50, 100, 1000, 1e4]
    list_marker = ["o", "v", "P", "X", "*", "^", 'D']
    
    # Plot for a = 0.99
    for m2_, fmt in zip(list_m2, list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": 0.99, "m2": m2_}
        dict_out = get_results(snr_dict_loaded, input_dict)
        ax1.errorbar(dict_out["m1"], dict_out["redshift_ref_median"], 
                     yerr=[dict_out["redshift_ref_median"] - dict_out["redshift_ref_m_sigma"], 
                           dict_out["redshift_ref_p_sigma"] - dict_out["redshift_ref_median"]], 
                     linestyle='none', capsize=7, fmt=fmt, label=f'm2={m2_}')
    ax1.set_ylabel("Redshift")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()
    ax1.set_title("a = 0.99")
    ax1.legend()
    
    # Plot for a = -0.99
    for m2_, fmt in zip(list_m2, list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": -0.99, "m2": m2_}
        dict_out = get_results(snr_dict_loaded, input_dict)
        ax2.errorbar(dict_out["m1"], dict_out["redshift_ref_median"], 
                     yerr=[dict_out["redshift_ref_median"] - dict_out["redshift_ref_m_sigma"], 
                           dict_out["redshift_ref_p_sigma"] - dict_out["redshift_ref_median"]], 
                     linestyle='none', capsize=7, fmt=fmt, label=f'm2={m2_}')
    ax2.set_ylabel("Redshift")
    ax2.set_xlabel("m1")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid()
    ax2.set_title("a = -0.99")
    ax2.legend()
    ax2.set_xlim(2e4, 2e7)
    plt.tight_layout()
    plt.savefig("redshift_vs_m1.png")
    
    input_dict = {"e_0": 0.0, "a": 0.99, "m2": 1.0, "m1": 1e6}
    dict_out = get_results(snr_dict_loaded, input_dict)
    plt.figure()
    plt.plot(dict_out["Tpl"], dict_out["snr"][:,2], color='k',alpha=0.1)
    plt.xlabel('Tobs')
    plt.ylabel('SNR')
    plt.savefig("snr_vs_Tobs.png")
    plt.show()