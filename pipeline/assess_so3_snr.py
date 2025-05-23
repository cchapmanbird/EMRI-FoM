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

# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [20.0, 25., 30.]

list_folders = sorted(glob.glob("./production_snr_*"))
print(list_folders)
list_results = []
# Assessment Process
for source in list_folders:
    print(f"Assessing science objectives for {source}...")
    # z_red = float(source.split("z=")[-1])
    m1_str = float(source.split("m1=")[-1].split("_")[0])
    # if m1_str not in [1e7, 1e6, 1e5, 1e4]:
    #     print(f"Skipping source {source}")
    #     continue
    Tpl = float(source.split("yr=")[-1].split("_")[0])
    redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
    detector_params = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["parameters"]
    source_params = detector_params.copy()
    source_params[0] = source_params[0] / (1 + redshift)
    source_params[1] = source_params[1] / (1 + redshift)
    # SNR assessment
    snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
    print(f"Mean SNR: {np.mean(snr_list)}")
    list_results.append({"m1":source_params[0], "m2":source_params[1], "snr":snr_list, "redshift":redshift, "dist":source_params[6], "Tpl": Tpl})
    # plot SNR histogram
    plt.figure()
    plt.hist(np.log10(snr_list), bins=30)
    plt.axvline(np.log10(thr_snr[0]), color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
    plt.axvline(np.log10(thr_snr[1]), color='orange', linestyle='--', label=f'Threshold {thr_snr[1]}')
    plt.axvline(np.log10(thr_snr[2]), color='green', linestyle='--', label=f'Threshold {thr_snr[2]}')
    plt.xlabel('Log10 SNR')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(f"{source}/snr_histogram.png")
    plt.close()
    # save snr distribution in folder
    np.savez(f"{source}/snr_distribution.npz", snr=snr_list)


# Prepare folders for Tpl == 0.5 and Tpl == 2.0
folders = {0.5: "Tpl_0.5", 2.0: "Tpl_2.0"}
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

for Tpl_val, folder in folders.items():
    # Collect m1, redshift, and mean SNR for each source with this Tpl
    m1_vals = np.asarray([np.log10(res["m1"]) for res in list_results if res["Tpl"] == Tpl_val])
    redshift_vals = np.asarray([res["redshift"] for res in list_results if res["Tpl"] == Tpl_val])
    mean_snr_vals = np.asarray([np.mean(res["snr"]) for res in list_results if res["Tpl"] == Tpl_val])
    snr_thresholds = [20.0, 25.0, 30.0]
    results_redshift_at_snr = {thr: {} for thr in snr_thresholds}

    unique_m1 = np.unique(m1_vals)
    for snr_threshold in snr_thresholds:
        for m1 in unique_m1:
            mask = m1_vals == m1
            z = redshift_vals[mask]
            snr = mean_snr_vals[mask]
            sort_idx = np.argsort(z)
            z_sorted = z[sort_idx]
            snr_sorted = snr[sort_idx]
            if np.any(snr_sorted >= snr_threshold) and np.any(snr_sorted <= snr_threshold):
                try:
                    z_at_snr = np.interp(np.log10(snr_threshold), np.log10(snr_sorted[::-1]), z_sorted[::-1])
                    results_redshift_at_snr[snr_threshold][m1] = z_at_snr
                    print(f"[Tpl={Tpl_val}] m1={10**m1:.1e}, redshift at SNR={snr_threshold}: {z_at_snr:.3f}")
                except Exception as e:
                    print(f"[Tpl={Tpl_val}] Interpolation failed for m1={10**m1:.1e}, SNR={snr_threshold}: {e}")
            else:
                print(f"[Tpl={Tpl_val}] m1={10**m1:.1e}: SNR does not cross threshold {snr_threshold}")

    # Plot redshift at each SNR threshold vs m1
    plt.figure()
    for snr_threshold in snr_thresholds:
        if results_redshift_at_snr[snr_threshold]:
            m1_plot = np.array(list(results_redshift_at_snr[snr_threshold].keys()))
            z_plot = np.array(list(results_redshift_at_snr[snr_threshold].values()))
            plt.plot(m1_plot, z_plot, 'o-', label=f'Redshift at SNR={snr_threshold}')
    plt.xlabel(r'$\log_{10} m_1$')
    plt.ylabel('Redshift at SNR threshold')
    plt.title(f'Redshift where SNR threshold is reached vs Mass (Tpl={Tpl_val})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/redshift_at_snr_thresholds_vs_mass.png")
    plt.close()

    plt.figure()
    import matplotlib.colors as mcolors
    levels = np.logspace(np.log10(20), np.log10(1000), 5)
    norm = mcolors.LogNorm(vmin=levels[0], vmax=mean_snr_vals.max() if len(mean_snr_vals) > 0 else 1000)
    if len(m1_vals) > 0 and len(redshift_vals) > 0 and len(mean_snr_vals) > 0:
        sc = plt.tricontourf(m1_vals, redshift_vals, mean_snr_vals, levels=levels, cmap='viridis', norm=norm)
        cbar = plt.colorbar(sc)
        cbar.set_label('SNR')
        cbar.set_ticks(levels)
        cbar.set_ticklabels([f"{l:.0f}" for l in levels])
        cbar.ax.set_yscale('log')
        plt.xlabel(r'$m_1 [M_\odot]$')
        plt.ylabel('Redshift')
        plt.ylim(redshift_vals.min(), 1.6)
        plt.tight_layout()
        plt.grid()
        plt.savefig(f"{folder}/Mass_redshift_snr_scatter.png")
        plt.close()

    plt.figure()
    if len(m1_vals) > 0 and len(redshift_vals) > 0 and len(mean_snr_vals) > 0:
        sc = plt.scatter(redshift_vals, mean_snr_vals, c=m1_vals, cmap='viridis', s=60)
        plt.axhline(thr_snr[0], color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
        plt.yscale('log')
        plt.xlabel('Redshift')
        plt.ylabel('SNR')
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$\log_{10} m_1$')
        plt.tight_layout()
        plt.savefig(f"{folder}/SNR_vs_redshift.png")
        plt.close()

    plt.figure()
    if len(m1_vals) > 0 and len(redshift_vals) > 0 and len(mean_snr_vals) > 0:
        sc = plt.scatter(m1_vals, mean_snr_vals, c=redshift_vals, cmap='viridis', s=60)
        plt.axhline(thr_snr[0], color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
        plt.yscale('log')
        plt.xlabel(r'$\log_{10} m_1$')
        plt.ylabel('SNR')
        cbar = plt.colorbar(sc)
        cbar.set_label("Redshift")
        plt.tight_layout()
        plt.savefig(f"{folder}/SNR_vs_mass.png")
        plt.close()

    # Write markdown table
    md_path = os.path.join(folder, "redshift_at_snr_thresholds.md")
    with open(md_path, "w") as f:
        f.write(f"# Redshift at SNR thresholds for Tpl={Tpl_val}\n\n")
        f.write("| log10(m1) | " + " | ".join([f"z@SNR={thr}" for thr in snr_thresholds]) + " |\n")
        f.write("|---" * (len(snr_thresholds)+1) + "|\n")
        for m1 in sorted(unique_m1):
            row = [f"{m1:.2f}"]
            for thr in snr_thresholds:
                z_val = results_redshift_at_snr[thr].get(m1, "")
                row.append(f"{z_val:.3f}" if z_val != "" else "")
            f.write("| " + " | ".join(row) + " |\n")
