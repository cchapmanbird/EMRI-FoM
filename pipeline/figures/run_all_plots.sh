#!/bin/bash
# Run all plotting scripts to generate figures or clean up generated plots
# Usage:
#   bash figures/run_all_plots.sh          # Generate all plots (default)
#   bash figures/run_all_plots.sh generate # Generate all plots
#   bash figures/run_all_plots.sh clean    # Remove all generated plots

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

# List of generated plot files
PLOT_FILES=(
    "figures/z_at_snr.png"
    "figures/snr_fom_ranges_m2_1_Tpl_0.25_prograde_retrograde.png"
    "figures/scatter_snr_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_relative_precision_m1_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_relative_precision_m2_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_relative_precision_a_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_relative_precision_dist_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_absolute_precision_a_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_absolute_precision_OmegaS_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_relative_precision_m1_det_m1_vs_m2_spin_a0.99.png"
    "figures/scatter_relative_precision_m2_det_m1_vs_m2_spin_a0.99.png"
    "figures/precision_e0_vs_e0_by_mass_ratio_and_m1.png"
    "figures/emri_trajectories_time_frequency.png"
    "figures/emri_trajectories_frequency_eccentricity.png"
    "figures/mbh_distance.png"
    "figures/best_source_sensitivity.png"
    "figures/emri_imri_masses_m1_m2.png"
    "figures/tidal_radius_normalized.png"
    "figures/tidal_radius_solar.png"
)

# List of plot scripts to run
PLOT_SCRIPTS=(
    "figures/plot_redshift_at_snr.py"
    "figures/plot_snr_fom_ranges.py"
    "figures/plot_scatter_precision_m1_m2.py"
    "figures/plot_precision_e0_vs_e0.py"
    "figures/plot_mbh_distance.py"
    "figures/plot_best_source_sensitivity.py"
    "figures/plot_emri_imri_masses.py"
    "figures/plot_emri_trajectories.py"
    "figures/plot_tidal_radius.py"
    "figures/plot_precision_vs_tpl.py"
)

# Function to generate all plots
generate_plots() {
    echo "=== Running all figure generation scripts ==="
    echo "Working directory: $PIPELINE_DIR"
    cd "$PIPELINE_DIR"

    for i in "${!PLOT_SCRIPTS[@]}"; do
        script="${PLOT_SCRIPTS[$i]}"
        echo ""
        echo "$((i+1)). Generating: $script..."
        python "$script"
    done

    echo ""
    echo "=== All plots generated successfully ==="
}

# Function to clean up all generated plots
clean_plots() {
    echo "=== Cleaning up generated plots ==="
    cd "$PIPELINE_DIR"
    
    for plot_file in "${PLOT_FILES[@]}"; do
        if [[ -f "$plot_file" ]]; then
            echo "Removing: $plot_file"
            rm "$plot_file"
        else
            echo "Not found (skipping): $plot_file"
        fi
    done
    
    echo ""
    echo "=== Cleanup complete ==="
}

# Main script logic
case "${1:-generate}" in
    generate)
        generate_plots
        ;;
    clean)
        clean_plots
        ;;
    *)
        echo "Usage: $0 {generate|clean}"
        echo "  generate - Generate all plots (default)"
        echo "  clean    - Remove all generated plots"
        exit 1
        ;;
esac
