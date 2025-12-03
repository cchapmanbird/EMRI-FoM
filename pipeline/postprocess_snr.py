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
    
    
    unique_m1 = sorted(set([source_dict[i]['m1'] for i in source_dict]))
    unique_m2 = sorted(set([source_dict[i]['m2'] for i in source_dict]))
    unique_a = sorted(set([source_dict[i]['a'] for i in source_dict]))
    unique_e0 = sorted(set([source_dict[i]['e_0'] for i in source_dict]))
    unique_Tpl = sorted(set([source_dict[i]['Tpl'] for i in source_dict]))
    # save z_ref_median in source_dict
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
    
    # https://arxiv.org/pdf/2404.00941
    masses_qpe = np.asarray([1.2, 0.55, 0.55, 3.1, 42.5, 1.8, 5.5, 0.595, 6.55, 88.0, 5.8]) * 1e6
    z_qpe     = np.asarray([0.0181, 0.0505, 0.0175, 0.024, 0.044, 0.0237, 0.042, 0.13, 0.0206, 0.0136, 0.0053])
    # Data extracted from Table EM_measure arXiv-2501.03252v2
    smbh_data = [
        {"name": "UGC 01032", "mass": 1.1, "redshift": 0.01678, "alternate_names": "Mrk 359"},
        {"name": "UGC 12163", "mass": 1.1, "redshift": 0.02468, "alternate_names": "Ark 564"},
        {"name": "Swift J2127.4+5654", "mass": 1.5, "redshift": 0.01400, "alternate_names": ""},
        {"name": "NGC 4253", "mass": 1.8, "redshift": 0.01293, "alternate_names": "UGC 07344, Mrk 766"},
        {"name": "NGC 4051", "mass": 1.91, "redshift": 0.00234, "alternate_names": "UGC 07030"},
        {"name": "NGC 1365", "mass": 2.0, "redshift": 0.00545, "alternate_names": ""},
        {"name": "1H0707-495", "mass": 2.3, "redshift": 0.04056, "alternate_names": ""},
        {"name": "MCG-6-30-15", "mass": 2.9, "redshift": 0.00749, "alternate_names": ""},
        {"name": "NGC 5506", "mass": 5.0, "redshift": 0.00608, "alternate_names": "Mrk 1376"},
        {"name": "IRAS13224-3809", "mass": 6.3, "redshift": 0.06579, "alternate_names": ""},
        {"name": "Ton S180", "mass": 8.1, "redshift": 0.06198, "alternate_names": ""},
    ]
    
    list_mass = np.asarray([item["mass"] for item in smbh_data]) * 1e6
    list_redshift = np.asarray([item["redshift"] for item in smbh_data])
    list_name = [item["name"] for item in smbh_data]

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
    ax1.set_xlabel("m1")
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
    plt.close('all')
    #################################
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    # Create colormap and normalization
    norm = Normalize(vmin=np.log10(1), vmax=np.log10(1e4))
    cmap = cm.get_cmap('viridis')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    list_m2 = [1, 5, 10, 50, 100, 1000, 1e4]
    list_marker = ["o", "v", "P", "X", "*", "^", 'D']
    
    # Plot for a = 0.99
    for m2_, fmt in zip(list_m2, list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": 0.99, "m2": m2_}
        dict_out = get_results(snr_dict_loaded, input_dict)
        color = cmap(norm(np.log10(m2_)))

        ax1.errorbar(dict_out["m1"], dict_out["redshift_ref_median"], 
                     yerr=[dict_out["redshift_ref_median"] - dict_out["redshift_ref_m_sigma"], 
                           dict_out["redshift_ref_p_sigma"] - dict_out["redshift_ref_median"]], 
                     color=color,
                     linestyle='none', capsize=7, fmt=fmt)#, label=f'm2={m2_}')
    ax1.plot(masses_qpe, z_qpe, 'r*', markersize=12, label='QPE and QPO')
    ax1.plot(list_mass, list_redshift, 's', color='C0', markersize=8, label='arXiv-2501.03252v2')
    ax1.set_ylabel("Redshift")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()
    ax1.set_title("a = 0.99")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Plot for a = -0.99
    for m2_, fmt in zip(list_m2, list_marker):
        input_dict = {"Tpl": 0.25, "e_0": 0.0, "a": -0.99, "m2": m2_}
        dict_out = get_results(snr_dict_loaded, input_dict)
        color = cmap(norm(np.log10(m2_)))
        ax2.errorbar(dict_out["m1"], dict_out["redshift_ref_median"], 
                     yerr=[dict_out["redshift_ref_median"] - dict_out["redshift_ref_m_sigma"], 
                           dict_out["redshift_ref_p_sigma"] - dict_out["redshift_ref_median"]], 
                     color=color,
                     linestyle='none', capsize=7, fmt=fmt)#, label=f'm2={m2_}')
    ax2.set_ylabel("Redshift")
    ax2.plot(masses_qpe, z_qpe, 'r*', markersize=12)
    ax2.plot(list_mass, list_redshift, 's', color='C0', markersize=8, label='arXiv-2501.03252v2')
    ax2.set_xlabel("m1")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid()
    ax2.set_title("a = -0.99")
    # ax2.legend()
    ax2.set_xlim(2e4, 2e7)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], label='log10(m2)', pad=-0.1, shrink=0.7)
    
    
    with open("lvk_gw_events.json", "r") as f:
        lvk_gw_events = json.load(f)
        

    min_lvk = np.min([min(lvk_gw_events['primary_mass']), min(lvk_gw_events['secondary_mass'])])
    max_lvk = np.max([max(lvk_gw_events['primary_mass']), max(lvk_gw_events['secondary_mass'])])
    # print(min_lvk, max_lvk)
    # for el in lvk_gw_events['primary_mass']+lvk_gw_events['secondary_mass']:
    #     plt.axhline(el, color='grey', alpha=0.1)

    # Add ticks at the m2 values
    list_ = lvk_gw_events['primary_mass'] + lvk_gw_events['secondary_mass']
    for m_ in [min(list_), max(list_)]:
        cbar.ax.axhline(norm(np.log10(m_)), color='grey', linewidth=2.0, alpha=0.7)
    # for m_ in list_m2:
    #     cbar.ax.axhline(norm(np.log10(m_)**0.999), color='k', linewidth=2.0, alpha=0.7)
    
    
    plt.tight_layout()
    plt.savefig("redshift_vs_m1.png")
    plt.show()
    breakpoint()
    # plt.close('all')
    
    plt.figure()
    for m2 in [1, 10, 100, 1000]:
        input_dict = {"e_0": 0.0, "a": 0.99, "m2": m2, "m1": 1e6}
        dict_out = get_results(snr_dict_loaded, input_dict)
        plt.semilogy(dict_out["Tpl"], dict_out["snr_at_z_ref_median"], '-o',alpha=0.9, label=f'{m2}')
    plt.xlabel('Tobs')
    plt.ylabel('SNR')
    plt.legend()
    plt.savefig("snr_vs_Tobs.png")
    plt.show()
    
    
    T_obs = 1/12
    dt = 1.0
    sig = np.ones(int(T_obs*86400*365/dt))
    taper_duration = 3600 * 6.
    taper_length = int(2 * taper_duration / dt)
    hann = np.hanning(taper_length)
    plt.figure()
    plt.plot(sig)
    sig_tapered = sig.copy()
    sig_tapered[:int(taper_length/2)] *= hann[:int(taper_length/2)]
    sig_tapered[-int(taper_length/2):] *= hann[-int(taper_length/2):]
    plt.plot(sig_tapered)
    plt.show()
    
    # input_dict = {"a": 0.99, "m2": 1.0, "m1": 1e6}
    # dict_out = get_results(snr_dict_loaded, input_dict)
    # plt.figure()
    # plt.plot(dict_out["Tpl"], dict_out["snr"][:,2], color='k',alpha=0.1)
    # plt.xlabel('Tobs')
    # plt.ylabel('SNR')
    # plt.savefig("snr_vs_Tobs.png")
    # plt.show()