#!/usr/bin/env python
"""
Polar plots: relative precision for m1 and spin a (two panels)

Generates a side-by-side polar figure showing `relative_precision_m1`
and `relative_precision_a` using the same inference HDF5 files as
plot_precision_polar.py.
# python plot_precision_OmegaS_dist.py; python plot_precision_m1_a.py
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
from math import log10, floor

np.random.seed(42)

# Ensure we run from pipeline directory to find inference_*/ files
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(script_dir)
sys.path.insert(0, pipeline_dir)
os.chdir(pipeline_dir)

try:
    plt.style.use('physrev.mplstyle')
except Exception:
    pass

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

# Config
spin_a = 0.99
tpl_val = 0.25
degradation = 1.0
tolerance = 1e-6
selected_run_type = 'circular'

title_map = {
    "relative_precision_m1": r"Primary Mass Precision $\sigma_{m_1}/m_1$",
    "relative_precision_a": r"Spin Precision $\sigma_{a}/a$",
}

custom_r_grids = {
    "relative_precision_a": [0.01, 0.001, 0.0001, 0.00001],
    "relative_precision_m1": [0.001,0.01, 0.1, 1.0],
}

def format_sigfigs(v, n=2):
    if v == 1.0:
        return "1"
    if v == 5.0:
        return "5"
    if v == 10.0:
        return "10"
    if v == 50.0:
        return "50"
    if v == 0:
        return '0'
    magnitude = floor(log10(abs(v)))
    str_out = f'{v/(10**(magnitude)):.0f}' + fr'$\times 10^{{{magnitude}}}$'
    if v / (10**(magnitude)) == 1.0:
        str_out = rf'$10^{{{magnitude}}}$'
    return str_out

# Load inference files
inference_files = sorted(glob.glob('inference_*/inference.h5'))
inference_metadata = {}
inference_precision_data = {}

for inf_file in inference_files:
    try:
        source_id = int(inf_file.split('_')[1].split('/')[0])
    except Exception:
        continue
    with h5py.File(inf_file, 'r') as f:
        for run_type in ['circular', 'eccentric']:
            if run_type not in f.keys():
                continue
            run_group = f[run_type]
            key = (source_id, run_type)
            inference_metadata[key] = {
                'm1': float(np.round(run_group['m1'][()], decimals=5)),
                'm2': float(np.round(run_group['m2'][()], decimals=5)),
                'a': float(run_group['a'][()]),
                'T': float(np.round(run_group['Tpl'][()], decimals=5)),
                'redshift': float(run_group['redshift'][()]),
                'run_type': run_type,
            }
            detector_precision = run_group['detector_measurement_precision'][()]
            source_precision = run_group['source_measurement_precision'][()]
            param_names = run_group['param_names'][()]
            param_names = np.array(param_names, dtype=str).tolist()
            inference_metadata[key].update({"param_names": param_names})

            for name in param_names:
                if name == 'M':
                    inference_precision_data.setdefault(key, {})
                    inference_precision_data[key].update({
                        "relative_precision_m1": source_precision[:, param_names.index(name)] / inference_metadata[key]['m1']
                    })
                elif name == 'a':
                    inference_precision_data.setdefault(key, {})
                    inference_precision_data[key].update({
                        "relative_precision_a": detector_precision[:, param_names.index(name)] / inference_metadata[key]['a']
                    })


# --- Stack plots vertically and use a single shared legend ---
metrics = ['relative_precision_m1', 'relative_precision_a']

# Build figure with two polar axes stacked vertically
fig, axs = plt.subplots(2, 1, figsize=(3.25, 5), subplot_kw={'polar': True})

# Collect all m2 values for unified legend
all_m2_set = set()
precision_by_m2_all = []
for metric in metrics:
    precision_by_m2 = {}
    all_m1_vals = set()
    for src_key in sorted(inference_metadata.keys()):
        source_id, run_type = src_key
        if run_type != selected_run_type:
            continue
        if abs(inference_metadata[src_key]['a'] - spin_a) > tolerance or abs(inference_metadata[src_key]['T'] - tpl_val) > tolerance:
            continue
        m1 = inference_metadata[src_key]['m1']
        m2 = inference_metadata[src_key]['m2']
        if metric not in inference_precision_data.get(src_key, {}):
            continue
        precision_array = inference_precision_data[src_key][metric]
        precision_median = np.median(np.abs(precision_array))
        precision_deg = precision_median * np.sqrt(degradation)
        all_m1_vals.add(m1)
        precision_by_m2.setdefault(m2, {'m1': [], 'precision': []})
        precision_by_m2[m2]['m1'].append(m1)
        precision_by_m2[m2]['precision'].append(precision_deg)
        all_m2_set.add(m2)
    precision_by_m2_all.append((precision_by_m2, all_m1_vals))

m2_list = sorted(all_m2_set)
colors = plt.cm.tab20(np.linspace(0, 1, max(len(m2_list), 1)))
m2_color_dict = {m2: colors[idx] for idx, m2 in enumerate(m2_list)}

for ax_idx, (ax, metric, (precision_by_m2, all_m1_vals)) in enumerate(zip(axs.flatten(), metrics, precision_by_m2_all)):
    if not precision_by_m2:
        ax.set_axis_off()
        continue
    m1_list = sorted(all_m1_vals)
    num_m1 = len(m1_list)
    m1_to_sector = {m1_val: (i * (2 * np.pi / num_m1), (i + 1) * (2 * np.pi / num_m1)) for i, m1_val in enumerate(m1_list)}
    for m2 in m2_list:
        if m2 not in precision_by_m2:
            continue
        m1_vals = np.array(precision_by_m2[m2]['m1'])
        prec_vals = np.array(precision_by_m2[m2]['precision'])
        for i, m1_val in enumerate(m1_vals):
            if m1_val not in m1_to_sector:
                continue
            theta_min, theta_max = m1_to_sector[m1_val]
            theta = np.linspace(theta_min, theta_max, 100)
            ax.plot(theta, prec_vals[i] * np.ones_like(theta), color=m2_color_dict[m2], linewidth=2, alpha=0.8, linestyle='-')
    ax.set_rscale('log')
    ax.set_rlabel_position(90)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(False)
    # Remove angular (theta) labels and ticks
    ax.set_xticks([])
    ax.set_xticklabels([])
    # radial grid
    r_grids = custom_r_grids.get(metric, [])
    for r in r_grids:
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot(theta, np.full_like(theta, r), color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
        # Use scientific notation for grid label
        ax.text(np.pi / 2 + 0.6, r*1.8, format_sigfigs(r), ha='left', va='bottom', fontsize=6, color='gray')
    ax.spines['polar'].set_visible(False)
    # m1 labels at outer radius
    theta_mid_deg = [((i + 0.5) * (360 / num_m1)) for i in range(num_m1)]
    r_label = ax.get_ylim()[1] * 1.7
    for i, m1_val in enumerate(m1_list):
        angle_rad = np.deg2rad(theta_mid_deg[i])
        ax.text(angle_rad, r_label, format_sigfigs(m1_val), ha='center', va='center', fontsize=9,
                rotation=(angle_rad % np.pi) * 180 / np.pi - 90, rotation_mode='anchor')
    # Add anticlockwise arrow to the first plot only
    if ax_idx == 0 and num_m1 > 1:
        # Arrow parameters
        arrow_r = r_label * 1.05
        theta_start = np.deg2rad(theta_mid_deg[0]*1.5)
        theta_end = np.deg2rad(theta_mid_deg[1]*0.8)
        # Draw a curved arrow from first to second m1 label (anticlockwise)
        ax.annotate('',
            xy=(theta_end, arrow_r),
            xytext=(theta_start, arrow_r),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.1", alpha=0.4),
            annotation_clip=False)
        # Optionally add a label
        ax.text((theta_start+theta_end)/2, arrow_r*1.7, 'increasing $m_1 [M_\odot]$', ha='center', va='center', fontsize=8)
    ax.set_title(title_map.get(metric, metric), pad=20, fontsize=11)

# Unified legend at the bottom
legend_elements = [Line2D([0], [0], color=m2_color_dict[m2], lw=2, label=format_sigfigs(m2)) for m2 in m2_list]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(len(m2_list), 4), title=r'$m_2 [M_\odot]$')

plt.subplots_adjust(hspace=0.25, bottom=0.13, top=0.95)
outname = f'precision_m1_a_a_{spin_a}_tpl_{tpl_val}.png'
plt.savefig(os.path.join(script_dir, outname), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved figure: figures/{outname}")
