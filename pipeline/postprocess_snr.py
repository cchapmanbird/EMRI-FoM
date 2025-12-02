import h5py
import numpy as np
import glob
import json

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


def get_results(input_dict, quantile = 0.68):
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
    return_dict = {"m1": m1, "m2": m2, "a": a, "Tpl": Tpl, "e_0": e0 , "e_f": ef, 
                   "snr": snr,
                   "snr_ref": snr[:,3],
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
    
    import matplotlib.pyplot as plt
    
    input_dict = {"Tpl": 0.25, "e_0": 0.0}
    plt.figure()
    plt.loglog(get_results(input_dict)["m1"],get_results(input_dict)["m2"], 'o')
    plt.xlabel("m1")
    plt.ylabel("m2")
    plt.savefig("test_plot.png")
    
    plt.figure()
    list_m2 = [1, 5, 10, 50, 100, 1000, 1e4]
    # list_m2 = [10]
    list_marker = ["o", "v", "P", "X", "*", "^", 'D']
    for m2_,fmt in zip(list_m2,list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": 0.99, "m2": m2_}
        dict_out = get_results(input_dict)
        plt.errorbar(dict_out["m1"],dict_out["snr_median"][:,2],yerr=[dict_out["snr_m_sigma"][:,2], dict_out["snr_p_sigma"][:,2]], 
                    linestyle='none',capsize=7, fmt=fmt)
    plt.ylabel("SNR")
    plt.xlabel("m1")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig("snr_vs_m1.png")
    
    input_dict = {"e_0": 0.0, "a": 0.99, "m2": 1.0, "m1": 1e6}
    dict_out = get_results(input_dict)
    plt.figure()
    plt.plot(dict_out["Tpl"], dict_out["snr"][:,2], color='k',alpha=0.1)
    plt.xlabel('Tobs')
    plt.ylabel('SNR')
    plt.savefig("snr_vs_Tobs.png")