#!/bin/bash
# Run all plotting scripts to generate figures
# Execute from the pipeline directory: bash figures/run_all_plots.sh

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Running all figure generation scripts ==="
echo "Working directory: $PIPELINE_DIR"
cd "$PIPELINE_DIR"

echo ""
echo "1. Generating: Redshift at SNR threshold plot..."
python figures/plot_redshift_at_snr.py

echo ""
echo "2. Generating: SNR FoM ranges plot..."
python figures/plot_snr_fom_ranges.py

echo ""
echo "3. Generating: Scatter precision m1 vs m2 plots..."
python figures/plot_scatter_precision_m1_m2.py

echo ""
echo "4. Generating: Precision e0 vs e0 plot..."
python figures/plot_precision_e0_vs_e0.py

echo ""
echo "=== All plots generated successfully ==="
