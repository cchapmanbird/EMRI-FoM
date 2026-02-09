"""
Gradio App for EMRI Degradation Analysis
Reproduces the interactive widgets from degradation_analysis.ipynb

Two main visualizations:
1. Redshift vs m1 at fixed SNR threshold with EM observations overlay
2. Measurement precision vs m1 for parameter estimation analysis
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import h5py
import glob
import os

# Change to pipeline directory for data access
PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline')

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_detection_data():
    """Load all detection.h5 files from snr_* directories"""
    detection_files = sorted(glob.glob(os.path.join(PIPELINE_DIR, 'snr_*/detection.h5')))
    
    source_metadata = {}
    source_snr_data = {}
    
    for det_file in detection_files:
        # Extract source ID from directory name
        source_id = int(os.path.basename(os.path.dirname(det_file)).split('_')[1])
        
        with h5py.File(det_file, 'r') as f:
            source_metadata[source_id] = {
                'm1': float(np.round(f['m1'][()], decimals=5)),
                'm2': float(np.round(f['m2'][()], decimals=5)),
                'a': float(np.round(f['a'][()], decimals=5)),
                'p0': float(f['p0'][()]),
                'e0': float(f['e0'][()]),
                'T': float(np.round(f['Tpl'][()], decimals=5)),
            }
            
            snr_data = f['snr'][()]
            redshifts = f['redshift'][()]
            
            source_snr_data[source_id] = {}
            for z_idx, z_val in enumerate(redshifts):
                source_snr_data[source_id][float(z_val)] = snr_data[z_idx, :]
    
    return source_metadata, source_snr_data


def load_inference_data():
    """Load all inference.h5 files from inference_* directories"""
    inference_files = sorted(glob.glob(os.path.join(PIPELINE_DIR, 'inference_*/inference.h5')))
    
    inference_metadata = {}
    inference_precision_data = {}
    
    for inf_file in inference_files:
        source_id = int(os.path.basename(os.path.dirname(inf_file)).split('_')[1])
        
        with h5py.File(inf_file, 'r') as f:
            for run_type in ['circular', 'eccentric']:
                if run_type not in f.keys():
                    continue
                
                run_group = f[run_type]
                source_key = (source_id, run_type)
                
                inference_metadata[source_key] = {
                    'm1': float(np.round(run_group['m1'][()], decimals=5)),
                    'm2': float(np.round(run_group['m2'][()], decimals=5)),
                    'a': float(run_group['a'][()]),
                    'p0': float(run_group['p0'][()]),
                    'e0': float(run_group['e0'][()]),
                    'e_f': float(run_group['e_f'][()]),
                    'dist': float(run_group['dist'][()]),
                    'T': float(np.round(run_group['Tpl'][()], decimals=5)),
                    'redshift': float(run_group['redshift'][()]),
                    'snr': run_group['snr'][()],
                    'run_type': run_type,
                }
                
                param_names = run_group['param_names'][()]
                param_names = np.array(param_names, dtype=str).tolist()
                inference_metadata[source_key]['param_names'] = param_names
                
                detector_precision = run_group['detector_measurement_precision'][()]
                source_precision = run_group['source_measurement_precision'][()]
                
                inference_precision_data[source_key] = {}
                
                for ii, name in enumerate(param_names):
                    if name == 'M':
                        inference_precision_data[source_key].update({
                            "relative_precision_m1_det": detector_precision[:, ii] / 
                                                          (inference_metadata[source_key]['m1'] * 
                                                           (1 + inference_metadata[source_key]['redshift'])),
                            "relative_precision_m1": source_precision[:, ii] / inference_metadata[source_key]['m1']
                        })
                    elif name == 'mu':
                        inference_precision_data[source_key].update({
                            "relative_precision_m2_det": detector_precision[:, ii] / 
                                                          (inference_metadata[source_key]['m2'] * 
                                                           (1 + inference_metadata[source_key]['redshift'])),
                            "relative_precision_m2": source_precision[:, ii] / inference_metadata[source_key]['m2']
                        })
                    elif name == 'e0':
                        inference_precision_data[source_key].update({
                            "relative_precision_e0": detector_precision[:, ii] / inference_metadata[source_key]['e0']
                        })
                    elif name == 'a':
                        inference_precision_data[source_key].update({
                            "absolute_precision_a": detector_precision[:, ii],
                            "relative_precision_a": detector_precision[:, ii] / abs(inference_metadata[source_key]['a']) if inference_metadata[source_key]['a'] != 0 else detector_precision[:, ii]
                        })
                    elif name == 'dist':
                        inference_precision_data[source_key].update({
                            "relative_precision_dist": detector_precision[:, ii] / inference_metadata[source_key]['dist']
                        })
                    else:
                        inference_precision_data[source_key].update({
                            "absolute_precision_" + name: detector_precision[:, ii]
                        })
    
    return inference_metadata, inference_precision_data


def load_em_observations():
    """Load electromagnetic observation data"""
    # QPE and QPO data (https://arxiv.org/pdf/2404.00941)
    masses_qpe = np.asarray([1.2, 0.55, 0.55, 3.1, 42.5, 1.8, 5.5, 0.595, 6.55, 88.0, 5.8]) * 1e6
    z_qpe = np.asarray([0.0181, 0.0505, 0.0175, 0.024, 0.044, 0.0237, 0.042, 0.13, 0.0206, 0.0136, 0.0053])
    
    # AGN data from Table EM_measure arXiv-2501.03252v2
    smbh_data = [
        {"name": "UGC 01032", "mass": 1.1, "redshift": 0.01678},
        {"name": "UGC 12163", "mass": 1.1, "redshift": 0.02468},
        {"name": "Swift J2127.4+5654", "mass": 1.5, "redshift": 0.01400},
        {"name": "NGC 4253", "mass": 1.8, "redshift": 0.01293},
        {"name": "NGC 4051", "mass": 1.91, "redshift": 0.00234},
        {"name": "NGC 1365", "mass": 2.0, "redshift": 0.00545},
        {"name": "1H0707-495", "mass": 2.3, "redshift": 0.04056},
        {"name": "MCG-6-30-15", "mass": 2.9, "redshift": 0.00749},
        {"name": "NGC 5506", "mass": 5.0, "redshift": 0.00608},
        {"name": "IRAS13224-3809", "mass": 6.3, "redshift": 0.06579},
        {"name": "Ton S180", "mass": 8.1, "redshift": 0.06198},
    ]
    smbh_masses = np.array([item['mass'] for item in smbh_data]) * 1e6
    smbh_redshifts = np.array([item['redshift'] for item in smbh_data])
    
    # Try to load SDSS DR16Q Quasars
    sdss_file = os.path.join(PIPELINE_DIR, 'sdss_dr16q_quasars.h5')
    try:
        with h5py.File(sdss_file, 'r') as f:
            redshift_sdss = f['redshift'][:]
            log10massbh_sdss = f['log10massbh'][:]
            log10massbh_err_sdss = f['log10massbh_err'][:]
            relative_error_mass = log10massbh_err_sdss * np.log(10)
            mask = (log10massbh_sdss < 7.05) & (relative_error_mass < 0.5)
            redshift_sdss = redshift_sdss[mask]
            massbh_sdss = 10**log10massbh_sdss[mask]
    except:
        redshift_sdss = np.array([])
        massbh_sdss = np.array([])
    
    return {
        'qpe': {'masses': masses_qpe, 'redshifts': z_qpe},
        'agn': {'masses': smbh_masses, 'redshifts': smbh_redshifts},
        'sdss': {'masses': massbh_sdss, 'redshifts': redshift_sdss}
    }


def extract_parameter_values(source_metadata, source_snr_data):
    """Extract unique parameter values for dropdown menus"""
    Tpl_values = sorted(set(source_metadata[src]['T'] for src in source_metadata))
    a_values = sorted(set(source_metadata[src]['a'] for src in source_metadata))
    m1_values = sorted(set(source_metadata[src]['m1'] for src in source_metadata))
    m2_values = sorted(set(source_metadata[src]['m2'] for src in source_metadata))
    
    all_redshifts = set()
    for src_id in source_snr_data:
        all_redshifts.update(source_snr_data[src_id].keys())
    all_redshifts = sorted(all_redshifts)
    
    return Tpl_values, a_values, m1_values, m2_values, all_redshifts


def extract_inference_parameter_values(inference_metadata):
    """Extract unique parameter values from inference data"""
    Tpl_values_inf = sorted(set(inference_metadata[src]['T'] for src in inference_metadata))
    a_values_inf = sorted(set(inference_metadata[src]['a'] for src in inference_metadata))
    m1_values_inf = sorted(set(inference_metadata[src]['m1'] for src in inference_metadata))
    m2_values_inf = sorted(set(inference_metadata[src]['m2'] for src in inference_metadata))
    run_types = sorted(set(src[1] for src in inference_metadata.keys()))
    
    return Tpl_values_inf, a_values_inf, m1_values_inf, m2_values_inf, run_types


# ============================================================================
# Try to load data, use demo data if files not found
# ============================================================================

try:
    source_metadata, source_snr_data = load_detection_data()
    if len(source_metadata) == 0:
        raise FileNotFoundError("No detection files found")
    Tpl_values, a_values, m1_values, m2_values, all_redshifts = extract_parameter_values(
        source_metadata, source_snr_data
    )
    DATA_LOADED = True
    print(f"Loaded SNR data for {len(source_metadata)} sources")
except Exception as e:
    print(f"Could not load detection data: {e}")
    print("Using demo data instead")
    DATA_LOADED = False
    
    # Demo parameter values
    Tpl_values = [0.25, 0.5, 1.0, 2.0, 4.5]
    a_values = [-0.99, -0.5, 0.0, 0.5, 0.99]
    m1_values = [1e5, 5e5, 1e6, 5e6, 1e7]
    m2_values = [1.0, 5.0, 10.0, 50.0, 100.0]
    all_redshifts = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    # Create demo source data
    source_metadata = {}
    source_snr_data = {}
    src_id = 0
    for tpl in Tpl_values:
        for a in a_values:
            for m1 in m1_values:
                for m2 in m2_values:
                    source_metadata[src_id] = {
                        'm1': m1, 'm2': m2, 'a': a, 'p0': 10.0, 'e0': 0.1, 'T': tpl
                    }
                    source_snr_data[src_id] = {}
                    for z in all_redshifts:
                        base_snr = 100 * (m1/1e6)**0.5 * (m2/10)**0.3 * (1+a)**0.5 * (tpl)**0.3 / (z + 0.1)
                        source_snr_data[src_id][z] = np.random.lognormal(
                            np.log(base_snr), 0.3, 100
                        )
                    src_id += 1

# Load inference data
try:
    inference_metadata, inference_precision_data = load_inference_data()
    if len(inference_metadata) == 0:
        raise FileNotFoundError("No inference files found")
    Tpl_values_inf, a_values_inf, m1_values_inf, m2_values_inf, run_types = extract_inference_parameter_values(
        inference_metadata
    )
    INFERENCE_LOADED = True
    print(f"Loaded inference data for {len(inference_metadata)} source configurations")
except Exception as e:
    print(f"Could not load inference data: {e}")
    INFERENCE_LOADED = False
    Tpl_values_inf = Tpl_values
    a_values_inf = a_values
    m2_values_inf = m2_values
    run_types = ['circular', 'eccentric']
    inference_metadata = {}
    inference_precision_data = {}

# Load EM observations
em_data = load_em_observations()

# Statistical estimator
estimator = np.median

# Mapping from metric keys to LaTeX labels
ylabel_map = {
    "relative_precision_m1_det": r"$\sigma_{m_{1,\mathrm{det}}}/m_{1,\mathrm{det}}$",
    "relative_precision_m1": r"$\sigma_{m_{1}}/m_{1}$",
    "relative_precision_m2_det": r"$\sigma_{m_{2,\mathrm{det}}}/m_{2,\mathrm{det}}$",
    "relative_precision_m2": r"$\sigma_{m_{2}}/m_{2}$",
    "relative_precision_dist": r"$\sigma_{d_L}/d_L$",
    "relative_precision_e0": r"$\sigma_{e_0}/e_0$",
    "absolute_precision_a": r"$\sigma_{a}$",
    "relative_precision_a": r"$\sigma_{a}/a$",
}

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_redshift_at_snr(tpl_val, spin_a, snr_threshold, degradation, m2_filter, show_em):
    """
    Plot redshift reach vs primary mass m1 at fixed SNR threshold.
    Includes overlay of electromagnetic observations (QPE, AGN, SDSS Quasars).
    """
    tolerance = 1e-6
    matching_sources = []
    
    # Convert string inputs to numeric
    tpl_val = float(tpl_val)
    spin_a = float(spin_a)
    
    # Filter sources by Tpl and spin
    for src_idx in sorted(source_metadata.keys()):
        src_a = source_metadata[src_idx]['a']
        src_tpl = source_metadata[src_idx]['T']
        
        if abs(src_a - spin_a) < tolerance and abs(src_tpl - tpl_val) < tolerance:
            matching_sources.append(src_idx)
    
    if not matching_sources:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, f"No sources found for Tpl={tpl_val:.2f}, a={spin_a:.2f}",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Extract redshift reach data
    z_data = {}
    for src_idx in matching_sources:
        m1 = source_metadata[src_idx]['m1']
        m2 = source_metadata[src_idx]['m2']
        
        z_snr_dict = source_snr_data[src_idx]
        z_vals_list = sorted(z_snr_dict.keys())
        snr_median_per_z = []
        
        for z in z_vals_list:
            snr_array = z_snr_dict[z]
            snr_median_per_z.append(estimator(snr_array))
        
        snr_median_per_z = np.array(snr_median_per_z)
        z_vals_array = np.array(z_vals_list)
        
        if snr_threshold > np.max(snr_median_per_z):
            continue
        
        try:
            interp_func = interp1d(snr_median_per_z, z_vals_array, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
            z_at_snr = float(interp_func(snr_threshold))
            
            snr_median_per_z_deg = snr_median_per_z / np.sqrt(degradation)
            interp_func_deg = interp1d(snr_median_per_z_deg, z_vals_array, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
            z_at_snr_deg = float(interp_func_deg(snr_threshold))
        except:
            continue
        
        if m2 not in z_data:
            z_data[m2] = {'m1': [], 'z_orig': [], 'z_deg': []}
        z_data[m2]['m1'].append(m1)
        z_data[m2]['z_orig'].append(z_at_snr)
        z_data[m2]['z_deg'].append(z_at_snr_deg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(z_data), 1)))
    
    # Plot redshift vs m1 with degradation arrows
    for idx, m2 in enumerate(sorted(z_data.keys())):
        if m2_filter != 'All' and m2 != float(m2_filter.replace('m2 = ', '')):
            continue
        
        m1_vals = np.array(z_data[m2]['m1'])
        z_orig = np.array(z_data[m2]['z_orig'])
        z_deg = np.array(z_data[m2]['z_deg'])
        
        sort_idx = np.argsort(m1_vals)
        m1_sorted = m1_vals[sort_idx]
        z_orig_sorted = z_orig[sort_idx]
        z_deg_sorted = z_deg[sort_idx]
        
        ax.plot(m1_sorted, z_orig_sorted, 'o-', color=colors[idx],
                markersize=7, linewidth=1.5, label=f'{m2:.0f}', alpha=0.7)
        
        if degradation != 1.0:
            ax.plot(m1_sorted, z_deg_sorted, 's--', color=colors[idx],
                    markersize=6, linewidth=1.5, alpha=0.5)
            
            for i in range(len(m1_sorted)):
                ax.annotate('', xy=(m1_sorted[i], z_deg_sorted[i]),
                           xytext=(m1_sorted[i], z_orig_sorted[i]),
                           arrowprops=dict(arrowstyle='->', color=colors[idx],
                                         lw=1.5, alpha=0.6))
    
    # Overlay EM observations
    if show_em:
        legend_elements_em = []
        
        # QPE and QPO
        mask_qpe = em_data['qpe']['masses'] <= 1e7
        ax.plot(em_data['qpe']['masses'][mask_qpe], em_data['qpe']['redshifts'][mask_qpe], 
                'D', color='purple', alpha=0.5, markersize=8, label='QPE and QPO')
        legend_elements_em.append(
            Line2D([0], [0], marker='D', label='QPE and QPO', alpha=0.5, 
                   markerfacecolor='purple', markersize=8, linestyle='None', color='purple')
        )
        
        # AGN
        ax.plot(em_data['agn']['masses'], em_data['agn']['redshifts'], 
                'X', color='k', alpha=0.5, markersize=10, label='AGN')
        legend_elements_em.append(
            Line2D([0], [0], marker='X', label='AGN', alpha=0.5, 
                   markerfacecolor='k', markersize=10, linestyle='None', color='k')
        )
        
        # SDSS Quasars
        if len(em_data['sdss']['masses']) > 0:
            ax.plot(em_data['sdss']['masses'], em_data['sdss']['redshifts'], 
                    '.', color='blue', alpha=0.1, markersize=8, label='SDSS DR16Q', zorder=0)
            legend_elements_em.append(
                Line2D([0], [0], marker='.', label='SDSS Quasars', alpha=0.3, 
                       markerfacecolor='blue', markersize=8, linestyle='None', color='blue')
            )
        
        leg_em = ax.legend(handles=legend_elements_em, frameon=True, loc='lower left', 
                           title='EM Observations', framealpha=0.9)
        ax.add_artist(leg_em)
    
    ax.set_xlabel(r'Primary mass $m_1 [M_\odot]$', fontsize=14)
    ax.set_ylabel(f'Redshift at $\\mathrm{{SNR}}={int(snr_threshold):.0f}$', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-3, 5.)
    ax.set_xlim(4e4, 1.1e7)
    
    # Legend for secondary masses
    legend_elements_m2 = [Line2D([0], [0], marker='o', label=f'{m2:.0f}', markersize=7, 
                                  linestyle='-', color=colors[idx]) 
                          for idx, m2 in enumerate(sorted(z_data.keys())) 
                          if (m2_filter == 'All' or m2 == float(m2_filter.replace('m2 = ', '')))]
    
    if degradation != 1.0:
        legend_elements_m2.extend([
            Line2D([0], [0], marker='o', label='Current sensitivity',
                   markerfacecolor='gray', markersize=7, linestyle='-', color='gray'),
            Line2D([0], [0], marker='s', label=f'Degraded by d={degradation:.2f}',
                   markerfacecolor='gray', markersize=6, linestyle='--', color='gray')
        ])
    
    leg_m2 = ax.legend(handles=legend_elements_m2, bbox_to_anchor=(0.95, 1.15),
                       frameon=True, ncols=4, title=r'Secondary Mass $m_2 [M_\odot]$')
    
    plt.tight_layout()
    return fig


def plot_measurement_precision(spin_a, tpl_val, m2_filter, run_type_filter, precision_metric, degradation):
    """
    Plot measurement precision vs m1 for inference analysis.
    Shows effect of detector degradation on parameter estimation precision.
    """
    if not INFERENCE_LOADED:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "Inference data not available.\nPlease ensure inference_*/inference.h5 files exist.",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    tolerance = 1e-6
    precision_data = {}
    
    # Convert string inputs to numeric
    spin_a = float(spin_a)
    tpl_val = float(tpl_val)
    
    for src_key in sorted(inference_metadata.keys()):
        source_id, run_type = src_key
        
        # Filter by spin and Tpl
        src_a = inference_metadata[src_key]['a']
        src_tpl = inference_metadata[src_key]['T']
        if abs(src_a - spin_a) > tolerance or abs(src_tpl - tpl_val) > tolerance:
            continue
        
        # Filter by run_type if specified
        if run_type_filter != 'All' and run_type != run_type_filter.lower():
            continue
        
        m1 = inference_metadata[src_key]['m1']
        m2 = inference_metadata[src_key]['m2']
        
        # Check if this precision metric exists for this source
        if precision_metric not in inference_precision_data[src_key]:
            continue
        
        # Get precision array and compute median
        precision_array = inference_precision_data[src_key][precision_metric]
        precision_orig = estimator(precision_array)
        precision_deg = precision_orig * np.sqrt(degradation)
        
        data_key = (m2, run_type)
        
        if data_key not in precision_data:
            precision_data[data_key] = {'m1': [], 'precision_orig': [], 'precision_deg': []}
        
        precision_data[data_key]['m1'].append(m1)
        precision_data[data_key]['precision_orig'].append(precision_orig)
        precision_data[data_key]['precision_deg'].append(precision_deg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if not precision_data:
        ax.text(0.5, 0.5, f"No data found for spin a={spin_a:.2f}, Tpl={tpl_val:.2f}\nand metric {precision_metric}",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Get unique m2 values and create colormap
    all_m2_values_inf = sorted(set(m2 for m2, _ in precision_data.keys()))
    colors_inf = plt.cm.tab20(np.linspace(0, 1, len(all_m2_values_inf)))
    m2_to_color_inf = {m2: colors_inf[idx] for idx, m2 in enumerate(all_m2_values_inf)}
    
    # Plot measurement precision vs m1
    for (m2, run_type) in sorted(precision_data.keys()):
        if m2_filter != 'All' and m2 != float(m2_filter.replace('m2 = ', '')):
            continue
        
        m1_vals = np.array(precision_data[(m2, run_type)]['m1'])
        precision_orig = np.array(precision_data[(m2, run_type)]['precision_orig'])
        precision_deg = np.array(precision_data[(m2, run_type)]['precision_deg'])
        
        sort_idx = np.argsort(m1_vals)
        m1_sorted = m1_vals[sort_idx]
        precision_orig_sorted = precision_orig[sort_idx]
        precision_deg_sorted = precision_deg[sort_idx]
        
        color = m2_to_color_inf[m2]
        
        ax.plot(m1_sorted, precision_orig_sorted, 'o-', color=color,
                markersize=7, linewidth=1.5, label=f'{m2:.0f}', alpha=0.7)
        ax.plot(m1_sorted, precision_deg_sorted, 's--', color=color,
                markersize=6, linewidth=1.5, alpha=0.5)
        
        for i in range(len(m1_sorted)):
            ax.annotate('', xy=(m1_sorted[i], precision_deg_sorted[i]),
                       xytext=(m1_sorted[i], precision_orig_sorted[i]),
                       arrowprops=dict(arrowstyle='->', color=color,
                                     lw=1.5, alpha=0.6))
    
    ax.set_xlabel(r'Primary mass $m_1 [M_\odot]$', fontsize=14)
    ax.set_ylabel(ylabel_map.get(precision_metric, precision_metric), fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # First legend for m2 values
    leg1 = ax.legend(loc='upper center', ncols=4, frameon=True, 
                      title=r'Secondary mass $m_2 [M_\odot]$')
    ax.add_artist(leg1)
    
    # Second legend for line styles
    legend_elements = [
        Line2D([0], [0], marker='o', label='Current sensitivity',
               markerfacecolor='gray', markersize=7, linestyle='-', color='gray'),
        Line2D([0], [0], marker='s', label=f'Degraded by d={degradation:.2f}',
               markerfacecolor='gray', markersize=6, linestyle='--', color='gray')
    ]
    leg2 = ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    ax.add_artist(leg2)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Build Gradio Interface
# ============================================================================

# Create dropdown choices for SNR/redshift plot
tpl_choices = [str(t) for t in Tpl_values]
spin_choices = [str(a) for a in a_values]
m2_filter_choices = ['All'] + [f'm2 = {m2:.0f}' for m2 in sorted(m2_values)]

# Create dropdown choices for inference plot
tpl_choices_inf = [str(t) for t in Tpl_values_inf]
spin_choices_inf = [str(a) for a in a_values_inf]
m2_filter_choices_inf = ['All'] + [f'm2 = {m2:.0f}' for m2 in sorted(m2_values_inf)]
run_type_choices = ['All', 'Circular', 'Eccentric']
precision_metric_choices = [
    ('Relative m1 (source)', 'relative_precision_m1'),
    ('Relative m1 (detector)', 'relative_precision_m1_det'),
    ('Relative m2 (source)', 'relative_precision_m2'),
    ('Relative m2 (detector)', 'relative_precision_m2_det'),
    ('Relative distance', 'relative_precision_dist'),
    ('Relative e0', 'relative_precision_e0'),
    ('Absolute a', 'absolute_precision_a'),
    ('Relative a', 'relative_precision_a'),
]

# Set default values
default_tpl = tpl_choices[0] if tpl_choices else "0.25"
default_spin = spin_choices[-1] if spin_choices else "0.99"
default_tpl_inf = tpl_choices_inf[0] if tpl_choices_inf else "0.25"
default_spin_inf = spin_choices_inf[-1] if spin_choices_inf else "0.99"

with gr.Blocks(title="EMRI Degradation Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # EMRI Degradation Analysis Dashboard
    
    Interactive visualization showing how detector degradation affects EMRI detectability and parameter estimation.
    """)
    
    with gr.Tabs():
        # Tab 1: Redshift Reach
        with gr.TabItem("🔭 Redshift Reach"):
            gr.Markdown("""
            ### LISA Redshift Horizon vs Primary Mass
            
            Shows the maximum redshift at which EMRIs can be detected with a given SNR threshold.
            Includes overlay of electromagnetic observations (QPE, AGN, SDSS Quasars) for comparison.
            
            **Degradation Model**: Sₙ(f) → d × Sₙ(f), which translates to SNR_degraded = SNR_original / √d
            """)
            
            with gr.Row():
                tpl_dropdown_z = gr.Dropdown(
                    choices=tpl_choices,
                    value=default_tpl,
                    label="Mission Lifetime Tpl (years)"
                )
                spin_dropdown_z = gr.Dropdown(
                    choices=spin_choices,
                    value=spin_choices[-1] if spin_choices else "0.99",
                    label="Spin (a)"
                )
                m2_dropdown_z = gr.Dropdown(
                    choices=m2_filter_choices,
                    value='All',
                    label="Secondary Mass Filter"
                )
            
            with gr.Row():
                snr_slider_z = gr.Slider(
                    minimum=5,
                    maximum=100,
                    step=5,
                    value=30,
                    label="SNR Threshold"
                )
                degradation_slider_z = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    step=0.25,
                    value=1.0,
                    label="Degradation Factor (d)"
                )
                show_em_checkbox = gr.Checkbox(
                    value=True,
                    label="Show EM Observations"
                )
            
            # Generate initial plot with default values
            initial_plot_z = plot_redshift_at_snr(
                default_tpl, default_spin, 30, 1.0, 'All', True
            )
            
            plot_output_z = gr.Plot(label="Redshift Reach Plot", value=initial_plot_z)
            
            # Connect inputs to output
            inputs_z = [tpl_dropdown_z, spin_dropdown_z, snr_slider_z, 
                        degradation_slider_z, m2_dropdown_z, show_em_checkbox]
            
            for inp in inputs_z:
                inp.change(
                    fn=plot_redshift_at_snr,
                    inputs=inputs_z,
                    outputs=plot_output_z
                )
        
        # Tab 2: Measurement Precision
        with gr.TabItem("📐 Measurement Precision"):
            gr.Markdown("""
            ### Parameter Estimation Precision vs Primary Mass
            
            Shows how well LISA can measure EMRI source parameters using Fisher matrix analysis.
            Degradation increases measurement uncertainties as: σ_degraded = σ_original × √d
            
            **Available metrics**:
            - Mass precision (source or detector frame)
            - Distance precision
            - Spin precision
            - Eccentricity precision (eccentric orbits only)
            """)
            
            with gr.Row():
                spin_dropdown_p = gr.Dropdown(
                    choices=spin_choices_inf,
                    value=spin_choices_inf[-1] if spin_choices_inf else "0.99",
                    label="Spin (a)"
                )
                tpl_dropdown_p = gr.Dropdown(
                    choices=tpl_choices_inf,
                    value=tpl_choices_inf[0] if tpl_choices_inf else "0.25",
                    label="Mission Lifetime Tpl (years)"
                )
                m2_dropdown_p = gr.Dropdown(
                    choices=m2_filter_choices_inf,
                    value='All',
                    label="Secondary Mass Filter"
                )
            
            with gr.Row():
                run_type_dropdown = gr.Dropdown(
                    choices=run_type_choices,
                    value='Circular',
                    label="Orbit Type"
                )
                precision_metric_dropdown = gr.Dropdown(
                    choices=[label for label, _ in precision_metric_choices],
                    value='Relative m1 (source)',
                    label="Precision Metric"
                )
                degradation_slider_p = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    step=0.25,
                    value=1.0,
                    label="Degradation Factor (d)"
                )
            
            # Map display names to metric keys
            metric_name_to_key = {label: key for label, key in precision_metric_choices}
            
            # Generate initial plot with default values
            initial_plot_p = plot_measurement_precision(
                default_spin_inf, default_tpl_inf, 'All', 'Circular', 'relative_precision_m1', 1.0
            )
            
            plot_output_p = gr.Plot(label="Measurement Precision Plot", value=initial_plot_p)
            
            def plot_precision_wrapper(spin_a, tpl_val, m2_filter, run_type, metric_name, degradation):
                metric_key = metric_name_to_key.get(metric_name, 'relative_precision_m1')
                return plot_measurement_precision(spin_a, tpl_val, m2_filter, run_type, metric_key, degradation)
            
            inputs_p = [spin_dropdown_p, tpl_dropdown_p, m2_dropdown_p,
                        run_type_dropdown, precision_metric_dropdown, degradation_slider_p]
            
            for inp in inputs_p:
                inp.change(
                    fn=plot_precision_wrapper,
                    inputs=inputs_p,
                    outputs=plot_output_p
                )

if __name__ == "__main__":
    # Launch with share=True creates a temporary public URL (72 hours)
    # For permanent hosting, deploy to Hugging Face Spaces or a cloud server
    demo.launch(share=True)