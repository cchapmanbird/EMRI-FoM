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
    "figures/*.png"
    "figures/*.pdf"
    "figures/*.md"
)

# List of plot scripts to run
PLOT_SCRIPTS=(
    "figures/plot_emri_imri_masses.py" # Fig 1
    "figures/plot_emri_trajectories.py" # Fig 2
    "figures/plot_redshift_at_snr.py" # Fig 4
    "figures/plot_precision_m1_a.py" # Fig 5
    "figures/plot_precision_e0_vs_e0.py" # Fig 6
    "figures/plot_precision_OmegaS_dist.py" # Fig 7
    "figures/plot_precision_vs_tpl.py" # Fig 8
    "figures/plot_powerlaw.py" # Fig 9 - 11
    "figures/plot_redshift_horizon_polar.py" # Fig 10
    "figures/plot_redshift_horizon_polar_m2.py" # Fig 10
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
