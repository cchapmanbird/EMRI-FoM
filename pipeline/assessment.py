import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
with open(f"fom_sources_light.json", "r") as json_file:
    source_intr = json.load(json_file)

def load_snr_data(source_name, par):
    """Load SNR data from .npz files in the source directory."""
    filelist = glob.glob(f"{source_name}/{par}.npz")
    results = np.load(filelist[0])
    for f in results.files:
        return results[f]

def plot_snr_histogram(snr_data, source_name, thresholds):
    """Plot and save the SNR histogram."""
    if not snr_data:
        print(f"No SNR data found for {source_name}. Skipping...")
        return

    plt.figure()
    plt.hist(np.log10(snr_data), bins=30, color='blue', alpha=0.7, label='SNR Data')
    for i, threshold in enumerate(thresholds):
        plt.axvline(np.log10(threshold), color=['r', 'orange', 'green'][i], linestyle='--', label=f'Threshold {threshold}')
    plt.xlabel(r'$\log_{10}$SNR')
    plt.ylabel('Counts')
    info = f"m1={source_intr[source_name]["m1_source"]:.1e},m2={source_intr[source_name]["m2_source"]:.1e},z={source_intr[source_name]["redshift"]:.1f},a={source_intr[source_name]["a"]:.1e},ef={source_intr[source_name]["e_f"]:.1e}"
    plt.title(f"SNR Histogram for {source_name} \n {info}")
    plt.legend()
    plt.savefig(f"{source_name}/snr_histogram.png")
    plt.close()
    print(f"SNR histogram saved to {source_name}/snr_histogram.png")

def plot_snr_histograms_table(sources, thresholds, par="snr_distribution"):
    """Plot a table of SNR histograms for all sources."""
    num_sources = len(sources)
    if num_sources == 0:
        print("No sources found. Exiting...")
        return

    # Determine grid size for subplots
    cols = 2  # Number of columns in the grid
    rows = (num_sources + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for i, source_name in enumerate(sources):
        snr_data = load_snr_data(source_name, par)

        axes[i].hist(np.log10(snr_data), bins=30, color='blue', alpha=0.7, label='SNR Data')
        for j, threshold in enumerate(thresholds):
            axes[i].axvline(np.log10(threshold), color=['r', 'orange', 'green'][j], linestyle='--', label=f'Threshold {threshold}')
        axes[i].set_xlabel(r'$\log_{10}$SNR')
        axes[i].set_ylabel('Counts')
        info = f"m1={source_intr[source_name]['m1_source']:.1e}, m2={source_intr[source_name]['m2_source']:.1e}, z={source_intr[source_name]['redshift']:.1f}, a={source_intr[source_name]['a']:.1e}, ef={source_intr[source_name]['e_f']:.1e}"
        axes[i].set_title(f"{source_name}\n{info}")
        axes[i].legend()

    # Turn off unused subplots
    for j in range(num_sources, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
    plt.savefig("histograms_table" + par +".png")
    plt.close()
    print("Table of " + par +" histograms saved")

def main():
    # Define the sources and thresholds
    sources = sorted([d for d in glob.glob("Source_*") if os.path.isdir(d)])
    thresholds = [10, 20, 30]  # Replace with actual threshold values
    names = sorted([d.split("/")[-1].split(".npz")[0] for d in glob.glob("Source_1/*.npz")])
    print("Creating table of histograms...")
    for par in names:
        plot_snr_histograms_table(sources, thresholds, par=par)

if __name__ == "__main__":
    main()
