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
import matplotlib.colors as mcolors
from common import CosmoInterpolator
import h5py
cosmo = CosmoInterpolator()
# the following two lines define the thresholds for the science objectives
# threshold_SNR: threshold on SNR for the science objectives
thr_snr = [20.0, 25., 30.]

list_folders = sorted(glob.glob("./production_snr_m*"))
print(list_folders)
list_results = []
h5_path = "snr_results.h5"
# check if the HDF5 file already exists
if os.path.exists(h5_path):
    print(f"HDF5 file {h5_path} already exists.")
else:
    with h5py.File(h5_path, "w") as h5f:
        # Assessment Process
        for source in list_folders:
            print(f"Assessing science objectives for {source}...")
            m1_str = float(source.split("m1=")[-1].split("_")[0])
            Tobs = float(source.split("T=")[-1].split("_")[0])
            signed_spin = float(source.split("a=")[-1].split("_")[0])
            snr_file = sorted(glob.glob(f"{source}/*/snr.npz"))[0]
            data = np.load(snr_file)
            redshift = data["redshift"]
            detector_params = data["parameters"]
            source_params = detector_params.copy()
            source_params[0] = source_params[0] / (1 + redshift)
            source_params[1] = source_params[1] / (1 + redshift)
            snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
            print(f"Mean SNR: {np.mean(snr_list)}")
            result = {
                "m1": source_params[0],
                "m2": source_params[1],
                "a": signed_spin,
                "p0": source_params[3],
                "e0": source_params[4],
                "snr": snr_list,
                "redshift": redshift,
                "dist": source_params[6],
                "Tobs": Tobs
            }
            list_results.append(result)
            # Store in HDF5
            grp = h5f.create_group(source)
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    grp.create_dataset(k, data=v)
                else:
                    grp.attrs[k] = v
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

# Set spin and Tobs values directly
spin = -0.99
for spin in [-0.99, 0.0, 0.99]:
    Tobs_val = 1.0  # Set your desired Tobs value here

    folder = f"Tobs_{Tobs_val}_spin_{spin}"
    os.makedirs(folder, exist_ok=True)

    with h5py.File(h5_path, "r") as h5f:
        m1_vals = []
        redshift_vals = []
        mean_snr_vals = []
        # Iterate over groups in HDF5 file
        for source in h5f:
            grp = h5f[source]
            # Check Tobs and spin
            Tobs = grp.attrs.get("Tobs", None)
            a = grp.attrs.get("a", None)
            print(f"Processing source: {source}, Tobs={Tobs}, a={a}")
            if Tobs is not None and a is not None and Tobs == Tobs_val and a == spin:
                m1 = grp.attrs["m1"] if "m1" in grp.attrs else grp["m1"][()]
                redshift = grp.attrs["redshift"] if "redshift" in grp.attrs else grp["redshift"][()]
                snr = grp["snr"][()]
                m1_vals.append(np.log10(m1))
                redshift_vals.append(redshift)
                mean_snr_vals.append(np.mean(snr))
        m1_vals = np.asarray(m1_vals)
        redshift_vals = np.asarray(redshift_vals)
        mean_snr_vals = np.asarray(mean_snr_vals)
        snr_thresholds = np.linspace(10., 50., 5)
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
                        print(f"[Tobs={Tobs_val}] m1={10**m1:.1e}, redshift at SNR={snr_threshold}: {z_at_snr:.3f}")
                    except Exception as e:
                        print(f"[Tobs={Tobs_val}] Interpolation failed for m1={10**m1:.1e}, SNR={snr_threshold}: {e}")
                else:
                    print(f"[Tobs={Tobs_val}] m1={10**m1:.1e}: SNR does not cross threshold {snr_threshold}")

        # Plot redshift at each SNR threshold vs m1
        plt.figure()
        for snr_threshold in snr_thresholds:
            if results_redshift_at_snr[snr_threshold]:
                m1_plot = np.array(list(results_redshift_at_snr[snr_threshold].keys()))
                z_plot = np.array(list(results_redshift_at_snr[snr_threshold].values()))
                plt.plot(m1_plot, z_plot, 'o-', label=f'Redshift at SNR={snr_threshold}')
        plt.xlabel(r'$\log_{10} m_1$')
        plt.ylabel('Redshift at SNR threshold')
        plt.title(f'Redshift where SNR threshold is reached vs Mass (Tobs={Tobs_val})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder}/redshift_at_snr_thresholds_vs_mass.png")
        plt.close()
        
        plt.figure()
        levels = np.logspace(np.log10(20), np.log10(mean_snr_vals.max()), 5)
        norm = mcolors.LogNorm(vmin=levels[0], vmax=mean_snr_vals.max() if len(mean_snr_vals) > 0 else 1000)
        if len(m1_vals) > 0 and len(redshift_vals) > 0 and len(mean_snr_vals) > 0:
            sc = plt.tricontourf(m1_vals, redshift_vals, mean_snr_vals, levels=levels, cmap='viridis', norm=norm)
            cbar = plt.colorbar(sc)
            cbar.set_label('SNR')
            cbar.set_ticks(levels)
            cbar.set_ticklabels([f"{l:.0f}" for l in levels])
            cbar.ax.set_yscale('log')
            plt.yscale('log')
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
            # Add theoretical line: (1+z)**(5/6) / D_L(z)
            z_theory = np.linspace(redshift_vals.min(), redshift_vals.max(), 200)
            D_L = cosmo.get_luminosity_distance(z_theory)
            theory_curve = (1 + z_theory) ** (5 / 6) / D_L
            # Scale for visual comparison
            theory_curve_scaled = theory_curve * mean_snr_vals.mean() / theory_curve.mean()/10
            plt.plot(z_theory, theory_curve_scaled, 'k--', label=r'$(1+z)^{5/6}/D_L(z)$ (scaled)')
            plt.legend()
            plt.axhline(thr_snr[0], color='r', linestyle='--', label=f'Threshold {thr_snr[0]}')
            plt.yscale('log')
            plt.xscale('log')
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
            f.write(f"# Redshift at SNR thresholds for Tobs={Tobs_val}\n\n")
            f.write("| log10(m1) | " + " | ".join([f"z@SNR={thr}" for thr in snr_thresholds]) + " |\n")
            f.write("|---" * (len(snr_thresholds)+1) + "|\n")
            for m1 in sorted(unique_m1):
                row = [f"{m1:.2f}"]
                for thr in snr_thresholds:
                    z_val = results_redshift_at_snr[thr].get(m1, "")
                    row.append(f"{z_val:.3f}" if z_val != "" else "")
                f.write("| " + " | ".join(row) + " |\n")
