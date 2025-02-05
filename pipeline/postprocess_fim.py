import argparse
import numpy as np
import pandas as pd
import glob
from common import h

parser = argparse.ArgumentParser()

parser.add_argument("indir", help="Directory containing source parameters and covariance matrices")
parser.add_argument("--src_numb", help="Source type to analyze", type=int, default=0)
parser.add_argument("--src_draw", help="Source draw to focus on", type=int, default=None)

args = parser.parse_args()

# Load covariance matrices and parameters
cov = [np.load(el) for el in glob.glob(f"{args.indir}/source_{args.src_numb}_*/*cov.npy")]
params_list = [np.load(el) for el in glob.glob(f"{args.indir}/source_{args.src_numb}_*/*params.npy")]
params = np.array(params_list)

# Ensure params has shape (NDraw, NParams)
if params.ndim == 1:
    params = np.expand_dims(params, axis=0)

columns = [
    "M", "mu", "a", "p0", "e0", "Y0", "dist", "qS", "phiS", 
    "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0", "z",
    "Tobs", "SNR"
]

# If --src_draw is provided, focus on that specific draw
if args.src_draw is not None:
    print(f"Focusing on source draw {args.src_draw}")
    params = params[args.src_draw:args.src_draw+1, :]
    cov = [cov[args.src_draw]]

# Reshape the 2D array to 2D (flattening the first two dimensions)
reshaped_data = params.reshape(-1, params.shape[-1])

# Generate indices for individual source (j)
num_sources_per_type = params.shape[0]
individual_sources = np.arange(num_sources_per_type)

# Create DataFrame and add the indices for clarity
df = pd.DataFrame(reshaped_data, columns=columns)

pd.set_option("display.max_colwidth", None)  # No truncation
pd.set_option("display.width", 150)
pd.set_option("display.colheader_justify", "center")  

df.insert(0, "Source Index", individual_sources)

# Columns to display
display_columns = ["M", "mu", "a", "p0", "e0", "Y0", "dist", "z", "Tobs", "SNR"]

# Filter DataFrame to show only the specified columns
df_display = df[display_columns]

# Compute average values
df_avg = df_display.mean().reset_index()

print("\n Average detector frame parameters at start of observation + SNR:\n")
print(df_avg.to_string(index=False, float_format="%.2e"))
print("\n")

# Compute and display precisions for parameters
precision_columns = [
    "ΔM / M", "Δmu / mu", "Δa", "Δp0", "Δe0", "Δdist / dist ", "ΔΩ", "Δz", "ΔM_s / M_s", "Δmu_s / mu_s"
]

all_precisions = []

for j in range(params.shape[0]):  # Iterate over each draw
    cov_matrix = cov[j]
    source_args = params[j]

    abs_precision = np.sqrt(np.diag(cov_matrix))

    df_precision = np.zeros(10)

    df_precision[0:2] = abs_precision[0:2] / source_args[0:2]  # relative error in detector frame masses
    df_precision[2:5] = abs_precision[2:5]  # absolute error in spin, p_0, and eccentricity
    df_precision[5] = abs_precision[5] / source_args[5]  # relative error in luminosity distance

    df_precision[6] = 2.0 * np.pi * np.sin(source_args[7]) * np.sqrt(cov_matrix[6, 6] * cov_matrix[7, 7] - cov_matrix[6, 7] ** 2) * (180 / np.pi) ** 2  # Sky location error converted to degree^2

    df_precision[7] = abs_precision[5] / (source_args[5] / (1 + source_args[-3]) + (1 + source_args[-3]) / h(source_args[-3]))  # absolute error in redshift

    df_precision[8] = df_precision[0] + df_precision[7] / (1 + source_args[-3])  # relative error in source frame mass

    df_precision[9] = df_precision[1] + df_precision[7] / (1 + source_args[-3])  # relative error in source frame mass

    all_precisions.append(df_precision)

# Convert all_precisions to a NumPy array for easier manipulation
all_precisions = np.array(all_precisions)

# Compute mean and standard deviation of precisions
mean_precisions = np.mean(all_precisions, axis=0)
std_precisions = np.std(all_precisions, axis=0)

# Create a DataFrame to store the mean and standard deviation
df_precisions = pd.DataFrame({
    "Parameter": precision_columns,
    "Mean Precision": mean_precisions,
    "Standard Deviation": std_precisions
})

print("\nMean and standard deviation of precisions:\n")
print(df_precisions.to_string(index=False, float_format="%.2e"))