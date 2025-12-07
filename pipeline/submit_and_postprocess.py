#!/usr/bin/env python3
"""
Unified script to submit SLURM jobs and postprocess results into HDF5.
Combines slurm_submit.py and postprocess_snr_inference_so3.py functionality.

Usage:
    # Submit SNR jobs
    python submit_and_postprocess.py --mode snr [--test]
    
    # Submit PE jobs
    python submit_and_postprocess.py --mode pe [--test]
    
    # Postprocess collected results
    python submit_and_postprocess.py --mode postprocess --parallel --num-jobs 128 --h5-output retry_so3_results.h5
    
    # Combined: Submit jobs then postprocess
    python submit_and_postprocess.py --mode snr --postprocess-after
"""

import os
import sys
import subprocess
import json
import numpy as np
from pathlib import Path
import argparse
import glob
import time
import h5py
import matplotlib.pyplot as plt
from filelock import FileLock


# ============================================================================
# SUBMISSION FUNCTIONS
# ============================================================================

def submit_slurm_job(source_params, pipeline_script="pipeline.py", partition="gpu_a100_7c"):
    """
    Submit a single SLURM job for an EMRI source.
    
    Args:
        source_params (dict): Dictionary containing source parameters
        pipeline_script (str): Path to the pipeline.py script
        partition (str): SLURM partition to use
    """
    repo_name = source_params['repo']
    job_id = repo_name.replace('/', '_').replace(' ', '_')
    
    # Create job-specific script
    job_script = f"slurm_job_{job_id}.sh"
    
    # Build the python command with all arguments
    python_cmd = f"python {pipeline_script} \\\n"
    python_cmd += f"    --M {source_params['M']} \\\n"
    python_cmd += f"    --mu {source_params['mu']} \\\n"
    python_cmd += f"    --a {source_params['a']} \\\n"
    python_cmd += f"    --e_f {source_params['e_f']} \\\n"
    python_cmd += f"    --T {source_params['T']} \\\n"
    python_cmd += f"    --z {source_params['z']} \\\n"
    python_cmd += f"    --repo {source_params['repo']} \\\n"
    python_cmd += f"    --psd_file {source_params['psd_file']} \\\n"
    python_cmd += f"    --channels {source_params['channels']} \\\n"
    python_cmd += f"    --dt {source_params['dt']} \\\n"
    python_cmd += f"    --use_gpu \\\n"
    python_cmd += f"    --N_montecarlo {source_params['N_montecarlo']} \\\n"
    python_cmd += f"    --device {source_params['device']} \\\n"
    python_cmd += f"    --calculate_fisher {source_params['pe']}"
    
    # Add extra arguments if present
    extra_args = source_params.get('extra_args', '')
    if extra_args:
        python_cmd += f" \\\n    {extra_args}"
    
    # Generate SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH -G a100:1
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -e {repo_name}.err
#SBATCH -o {repo_name}.out
#SBATCH --job-name=EMRI_{job_id}
#SBATCH -t 24:00:00

# Change to pipeline directory
cd $HOME/GitHub/EMRI-FoM/pipeline/

# Run the pipeline with parameters using Singularity container
singularity exec --nv ../fom_final.sif {python_cmd}

echo "Job ended."
"""
    
    # Write script to file
    with open(job_script, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(job_script, 0o755)
    
    # Submit job
    cmd = ["sbatch", job_script]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id_slurm = result.stdout.strip().split()[-1]
        print(f"✓ Submitted job {job_id_slurm}: {repo_name}")
        
        # Clean up job script after submission
        os.remove(job_script)
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job for {repo_name}: {e}")
        print(f"  Error output: {e.stderr}")
        return None


def generate_snr_sources(test_mode=False, repo_root="production_snr_", psd_file="TDI2_AE_psd.npy"):
    """
    Generate source parameters for SNR calculations.
    Based on pipeline_snr.py logic.
    """
    Nmonte = 1 if test_mode else 100
    dev = 0
    channels = 'AE'
    include_foreground = True
    esaorbits = True
    tdi2 = True
    
    sources = []
    
    with open("so3_snr_sources.json", "r") as f:
        source_dict = json.load(f)
    
    for key, params in source_dict.items():
        for redshift in np.logspace(-3, 1, 10):
            m1 = params["m1"]
            m2 = params["m2"]
            a = params["a"]
            ef = params["e_f"]
            Tobs = params["Tpl"]
            dt = params["dt"]
            
            if ef != 0.0:
                continue
                
            if Tobs != 1.5 and Tobs != 4.5:
                psd_file = "TDI2_AE_psd_emri_background_1.5_yr.npy"
            else:
                psd_file = f"TDI2_AE_psd_emri_background_{Tobs}_yr.npy"
            
            psd_name = psd_file.replace('.npy', '')
            source_name = repo_root + key + '/' + f"m1={m1}_m2={m2}_a={a}_e_f={ef}_T={Tobs}_z={redshift}_{psd_name}"
            
            # Build extra_args
            extra_args = ""
            if include_foreground:
                extra_args += " --foreground"
            if esaorbits:
                extra_args += " --esaorbits"
            if tdi2:
                extra_args += " --tdi2"
            
            sources.append({
                "M": m1 * (1 + redshift),
                "mu": m2 * (1 + redshift),
                "a": a,
                "e_f": ef,
                "T": Tobs,
                "z": redshift,
                "repo": source_name,
                "psd_file": psd_file,
                "channels": channels,
                "dt": dt,
                "N_montecarlo": Nmonte,
                "device": dev,
                "pe": 0,
                "extra_args": extra_args.strip(),
            })

    if test_mode:
        sources = sources[:1]
        
    # Save sources to file
    sources_file = repo_root + "sources_snr.txt"
    with open(sources_file, "w") as f:
        for source in sources:
            f.write(f"{source}\n")
    
    print(f"Generated {len(sources)} SNR sources")
    return sources


def generate_pe_sources(test_mode=False, repo_root="production_inference_", psd_file="TDI2_AE_psd.npy"):
    """
    Generate source parameters for parameter estimation (Fisher matrix).
    Based on pipeline_pe.py logic.
    """
    Nmonte = 1 if test_mode else 100
    dev = 0
    channels = 'AE'
    include_foreground = True
    esaorbits = True
    tdi2 = True
    
    sources = []
    
    with open("so3_inference_sources_with_z_ref.json", "r") as f:
        source_dict = json.load(f)
    
    for key, params in source_dict.items():
        m1 = params["m1"]
        m2 = params["m2"]
        a = params["a"]
        ef = params["e_f"]
        Tobs = params["Tpl"]
        dt = params["dt"]
        z = params["z_ref_median"]
        
        if Tobs != 1.5 and Tobs != 4.5:
            psd_file = "TDI2_AE_psd_emri_background_1.5_yr.npy"
        else:
            psd_file = f"TDI2_AE_psd_emri_background_{Tobs}_yr.npy"
        
        psd_name = psd_file.replace('.npy', '')
        source_name = repo_root + key + '/' + f"m1={m1}_m2={m2}_a={a}_e_f={ef}_T={Tobs}_z={z}_{psd_name}"
        
        # Build extra_args
        extra_args = ""
        if include_foreground:
            extra_args += " --foreground"
        if esaorbits:
            extra_args += " --esaorbits"
        if tdi2:
            extra_args += " --tdi2"
        
        if z != -1.0:
            sources.append({
                "M": m1 * (1 + z),
                "mu": m2 * (1 + z),
                "a": a,
                "e_f": ef,
                "T": Tobs,
                "z": z,
                "repo": source_name,
                "psd_file": psd_file,
                "channels": channels,
                "dt": dt,
                "N_montecarlo": Nmonte,
                "device": dev,
                "pe": 1,
                "extra_args": extra_args.strip(),
            })
    
    if test_mode:
        sources = sources[:1]
    
    # Save sources to file
    sources_file = repo_root + "sources_pe.txt"
    with open(sources_file, "w") as f:
        for source in sources:
            f.write(f"{source}\n")
    
    print(f"Generated {len(sources)} PE sources")
    return sources


# ============================================================================
# POSTPROCESSING FUNCTIONS
# ============================================================================

def process_single_source(source, full_names=None):
    """
    Process a single source and return results dict for HDF5 storage.
    
    Args:
        source (str): Path to source folder
        full_names (np.array): Parameter names array
    
    Returns:
        dict: Results to be stored in HDF5
    """
    if full_names is None:
        full_names = np.array(['m1','m2','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0', 'A', 'nr'])
    
    result = {}
    
    try:
        Tpl = float(source.split("T=")[-1].split("_")[0])
        redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
        detector_params = np.asarray([np.load(el)["parameters"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
        e_f = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["e_f"]
        source_params = detector_params[0].copy()
        source_params[0] = source_params[0] / (1 + redshift)
        source_params[1] = source_params[1] / (1 + redshift)
        lum_dist = detector_params[:,6]
        sky_loc = detector_params[:,7:9]
        spin_loc = detector_params[:, 9:11]
        detector_params = detector_params[0]
        snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
        
        result = {
            "m1": source_params[0],
            "m2": source_params[1],
            "a": source_params[2]*source_params[5],
            "p0": source_params[3],
            "e0": source_params[4],
            "DL": source_params[6],
            "e_f": e_f,
            "Tpl": Tpl,
            "redshift": redshift,
            "lum_dist": lum_dist,
            "snr": snr_list,
            "sky_loc": sky_loc,
            "spin_loc": spin_loc,
        }
        
        # SNR plot
        plt.figure()
        plt.hist(snr_list, bins=30)
        plt.xlabel('SNR')
        plt.ylabel('Counts')
        plt.savefig(f"{source}/snr_histogram.png", dpi=300)
        plt.close()
        
        if "inference" in source:
            # Fisher matrices and covariances
            param_names = np.asarray([np.load(el)["names"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            detector_cov = np.asarray([np.load(el)["cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
            fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
            source_measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
            detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
            
            names = param_names[0].tolist()
            ind_sky = [names.index('qS'), names.index('phiS')]
            ind_volume = [names.index('dist'), names.index('qS'), names.index('phiS')]
            
            qS = fish_params[:, ind_sky[0]]
            Sigma = source_cov[:,ind_sky[0]:ind_sky[1]+1, ind_sky[0]:ind_sky[1]+1]
            err_sky_loc = 2 * np.pi * np.sin(qS) * np.sqrt(np.linalg.det(Sigma)) * (180.0 / np.pi) ** 2
            names.append("OmegaS")
            source_measurement_precision = np.hstack((source_measurement_precision, err_sky_loc[:, None]))
            detector_measurement_precision = np.hstack((detector_measurement_precision, err_sky_loc[:, None]))
            
            Sigma_V = source_cov[:,ind_volume[0]:ind_volume[2]+1, ind_volume[0]:ind_volume[2]+1]
            err_volume = (4/3) * np.pi * (fish_params[:,ind_volume[0]])**2 * np.sqrt(np.linalg.det(Sigma_V))
            names.append("DeltaV")
            source_measurement_precision = np.hstack((source_measurement_precision, err_volume[:, None]))
            detector_measurement_precision = np.hstack((detector_measurement_precision, err_volume[:, None]))
            
            result["error_source"] = source_measurement_precision
            result["error_detector"] = detector_measurement_precision
            result["error_names"] = names
        
        return result
    
    except Exception as e:
        print(f"  ✗ Error processing {source}: {e}")
        return None


def postprocess_parallel(h5_output="so3_results.h5", num_jobs=4):
    """
    Postprocess results using parallel jobs with file locking.
    
    Args:
        h5_output (str): Output HDF5 file path
        num_jobs (int): Number of parallel jobs to submit
    """
    job_script_base = "slurm_postprocess_job"
    
    # Create Python wrapper script that handles parallel writing
    wrapper_script = "postprocess_parallel_worker.py"
    with open(wrapper_script, 'w') as f:
        f.write('''
import sys
import glob
import numpy as np
import h5py
import os
from filelock import FileLock
import matplotlib.pyplot as plt

def process_single_source(source, full_names=None):
    """Process a single source and return results dict for HDF5 storage."""
    if full_names is None:
        full_names = np.array(['m1','m2','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0', 'A', 'nr'])
    
    result = {}
    
    try:
        Tpl = float(source.split("T=")[-1].split("_")[0])
        redshift = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["redshift"]
        detector_params = np.asarray([np.load(el)["parameters"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
        e_f = np.load(sorted(glob.glob(f"{source}/*/snr.npz"))[0])["e_f"]
        source_params = detector_params[0].copy()
        source_params[0] = source_params[0] / (1 + redshift)
        source_params[1] = source_params[1] / (1 + redshift)
        lum_dist = detector_params[:,6]
        sky_loc = detector_params[:,7:9]
        spin_loc = detector_params[:, 9:11]
        detector_params = detector_params[0]
        snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(f"{source}/*/snr.npz"))])
        
        result = {
            "m1": source_params[0],
            "m2": source_params[1],
            "a": source_params[2]*source_params[5],
            "p0": source_params[3],
            "e0": source_params[4],
            "DL": source_params[6],
            "e_f": e_f,
            "Tpl": Tpl,
            "redshift": redshift,
            "lum_dist": lum_dist,
            "snr": snr_list,
            "sky_loc": sky_loc,
            "spin_loc": spin_loc,
        }
        
        plt.figure()
        plt.hist(snr_list, bins=30)
        plt.xlabel('SNR')
        plt.ylabel('Counts')
        plt.savefig(f"{source}/snr_histogram.png", dpi=300)
        plt.close()
        
        if "inference" in source:
            param_names = np.asarray([np.load(el)["names"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            detector_cov = np.asarray([np.load(el)["cov"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            fish_params = np.asarray([np.load(el)["fisher_params"] for el in sorted(glob.glob(f"{source}/*/results.npz"))])
            fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
            fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
            source_measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
            detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
            
            names = param_names[0].tolist()
            ind_sky = [names.index('qS'), names.index('phiS')]
            ind_volume = [names.index('dist'), names.index('qS'), names.index('phiS')]
            
            qS = fish_params[:, ind_sky[0]]
            Sigma = source_cov[:,ind_sky[0]:ind_sky[1]+1, ind_sky[0]:ind_sky[1]+1]
            err_sky_loc = 2 * np.pi * np.sin(qS) * np.sqrt(np.linalg.det(Sigma)) * (180.0 / np.pi) ** 2
            names.append("OmegaS")
            source_measurement_precision = np.hstack((source_measurement_precision, err_sky_loc[:, None]))
            detector_measurement_precision = np.hstack((detector_measurement_precision, err_sky_loc[:, None]))
            
            Sigma_V = source_cov[:,ind_volume[0]:ind_volume[2]+1, ind_volume[0]:ind_volume[2]+1]
            err_volume = (4/3) * np.pi * (fish_params[:,ind_volume[0]])**2 * np.sqrt(np.linalg.det(Sigma_V))
            names.append("DeltaV")
            source_measurement_precision = np.hstack((source_measurement_precision, err_volume[:, None]))
            detector_measurement_precision = np.hstack((detector_measurement_precision, err_volume[:, None]))
            
            result["error_source"] = source_measurement_precision
            result["error_detector"] = detector_measurement_precision
            result["error_names"] = names
        
        return result
    
    except Exception as e:
        print(f"  ✗ Error processing {source}: {e}")
        return None

job_id = int(sys.argv[1])
num_jobs = int(sys.argv[2])
h5_output = sys.argv[3]

# Get all folders to process
list_folders = sorted(glob.glob("./production_inference*/*/")) + sorted(glob.glob("./production_snr*/*/"))

# Distribute folders across jobs
folders_per_job = len(list_folders) // num_jobs
start_idx = job_id * folders_per_job
end_idx = start_idx + folders_per_job if job_id < num_jobs - 1 else len(list_folders)
assigned_folders = list_folders[start_idx:end_idx]

print(f"Job {job_id}: Processing folders {start_idx} to {end_idx} ({len(assigned_folders)} folders)")

# Use FileLock for safe parallel HDF5 writes
lock_file = h5_output + ".lock"
lock = FileLock(lock_file, timeout=300)

for source in assigned_folders:
    if os.path.isdir(source) is False:
        continue
    
    print(f"[Job {job_id}] Processing {source}")
    result = process_single_source(source)
    
    if result is None:
        continue
    
    # Write to HDF5 with lock
    with lock:
        with h5py.File(h5_output, "a") as h5f:
            if source not in h5f:
                grp = h5f.create_group(source)
                for k, v in result.items():
                    grp.create_dataset(k, data=v)
    
    print(f"[Job {job_id}] Saved {source} to HDF5")

print(f"Job {job_id} completed")
''')
    
    # Generate SLURM job scripts
    job_ids = []
    for job_num in range(num_jobs):
        job_script = f"{job_script_base}_{job_num}.sh"
        
        script_content = f"""#!/bin/bash
#SBATCH -p normal
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -e postprocess_job_{job_num}.err
#SBATCH -o postprocess_job_{job_num}.out
#SBATCH --job-name=EMRI_postprocess_{job_num}
#SBATCH -t 12:00:00

cd $HOME/GitHub/EMRI-FoM/pipeline/
source fom_venv/bin/activate

# Run postprocessing with job number
python postprocess_parallel_worker.py {job_num} {num_jobs} {h5_output}

echo "Postprocess job {job_num} completed"
"""
        
        with open(job_script, 'w') as f:
            f.write(script_content)
        
        os.chmod(job_script, 0o755)
        
        # Submit job
        try:
            result = subprocess.run(["sbatch", job_script], capture_output=True, text=True, check=True)
            job_id_slurm = result.stdout.strip().split()[-1]
            job_ids.append(job_id_slurm)
            print(f"✓ Submitted postprocess job {job_num}: {job_id_slurm}")
            os.remove(job_script)
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to submit job {job_num}: {e}")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
    
    return job_ids


def postprocess_sequential(h5_output="so3_results.h5"):
    """
    Postprocess results sequentially (in current process).
    Useful for immediate postprocessing after jobs complete.
    
    Args:
        h5_output (str): Output HDF5 file path
    """
    full_names = np.array(['m1','m2','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0', 'A', 'nr'])
    list_folders = sorted(glob.glob("./production_inference*/*/")) + sorted(glob.glob("./production_snr*/*/"))
    
    print(f"\nStarting sequential postprocessing of {len(list_folders)} sources...")
    
    # Check if HDF5 file already exists
    if os.path.exists(h5_path):
        print(f"HDF5 file {h5_path} already exists. Appending...")
        mode = "a"
    else:
        print(f"Creating new HDF5 file {h5_path}...")
        mode = "w"
    
    processed = 0
    skipped = 0
    
    with h5py.File(h5_output, mode) as h5f:
        for i, source in enumerate(list_folders):
            if os.path.isdir(source) is False:
                continue
            
            # Skip if already processed
            if source in h5f:
                print(f"  [{i+1}/{len(list_folders)}] Skipping (already processed): {source}")
                skipped += 1
                continue
            
            print(f"  [{i+1}/{len(list_folders)}] Processing: {source}")
            result = process_single_source(source, full_names)
            
            if result is None:
                continue
            
            # Store in HDF5
            grp = h5f.create_group(source)
            for k, v in result.items():
                grp.create_dataset(k, data=v)
            
            processed += 1
    
    print(f"\n{'='*60}")
    print(f"Postprocessing complete!")
    print(f"  Processed: {processed} new sources")
    print(f"  Skipped: {skipped} sources (already in HDF5)")
    print(f"  Results saved to: {h5_output}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs and postprocess results for EMRI FoM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit SNR calculation jobs (production mode)
  python submit_and_postprocess.py --mode snr
  
  # Submit PE calculation jobs (test mode)
  python submit_and_postprocess.py --mode pe --test
  
  # Postprocess results sequentially
  python submit_and_postprocess.py --mode postprocess
  
  # Postprocess results with parallel SLURM jobs
  python submit_and_postprocess.py --mode postprocess --parallel --num-jobs 4
  
  # Check current queue status
  python submit_and_postprocess.py --check-queue
        """
    )
    
    parser.add_argument("--mode", choices=["snr", "pe", "postprocess"], 
                       help="Pipeline mode: 'snr' for SNR calculations, 'pe' for parameter estimation, 'postprocess' for HDF5 postprocessing")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode (1 source, 1 Monte Carlo)")
    parser.add_argument("--partition", type=str, default="gpu_a100_22c",
                       help="SLURM partition to use for job submission (default: gpu_a100_22c)")
    parser.add_argument("--psd", type=str, 
                       choices=["TDI2_AE_psd_emri_background_1.5_yr.npy", "TDI2_AE_psd_emri_background_4.5_yr.npy"],
                       default="TDI2_AE_psd_emri_background_4.5_yr.npy",
                       help="PSD file to use")
    
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel SLURM jobs for postprocessing (with --mode postprocess)")
    parser.add_argument("--num-jobs", type=int, default=4,
                       help="Number of parallel postprocessing jobs (default: 4)")
    parser.add_argument("--h5-output", type=str, default="so3_results.h5",
                       help="Output HDF5 file path (default: so3_results.h5)")
    
    parser.add_argument("--check-queue", action="store_true",
                       help="Check current SLURM queue status")
    
    args = parser.parse_args()
    
    # Change to pipeline directory
    pipeline_dir = Path(__file__).parent
    os.chdir(pipeline_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Check queue status if requested
    if args.check_queue:
        try:
            result = subprocess.run(["squeue", "-u", os.getenv("USER")], 
                          capture_output=True, text=True, check=True)
            print("Current queue status:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to check queue: {e}")
        return
    
    # Postprocessing modes
    if args.mode == "postprocess":
        if args.parallel:
            print(f"Submitting {args.num_jobs} parallel postprocessing jobs...")
            job_ids = postprocess_parallel(h5_output=args.h5_output, num_jobs=args.num_jobs)
            print(f"\nSubmitted {len(job_ids)} postprocessing jobs")
            if job_ids:
                print(f"Monitor with: squeue -j {','.join(job_ids)}")
        else:
            postprocess_sequential(h5_output=args.h5_output)
        return
    
    # Job submission modes
    if not args.mode:
        parser.error("--mode is required unless using --check-queue")
    
    # Generate sources based on mode
    if args.mode == "snr":
        repo_root = "test_snr_" if args.test else "production_snr_"
        sources = generate_snr_sources(test_mode=args.test, repo_root=repo_root, psd_file=args.psd)
    else:  # pe mode
        repo_root = "test_pe_" if args.test else "production_inference_"
        sources = generate_pe_sources(test_mode=args.test, repo_root=repo_root)
    
    print(f"\nSubmitting {len(sources)} jobs in {args.mode} mode...")
    print(f"Partition: {args.partition}")
    if args.test:
        print("TEST MODE: Running with reduced parameters\n")
    
    # Submit all jobs
    submitted = 0
    failed = 0
    
    for source in sources:
        result = submit_slurm_job(source, partition=args.partition)
        if result:
            submitted += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Submitted: {submitted} jobs")
    print(f"  Failed: {failed} jobs")
    print(f"{'='*60}")
    
    if submitted > 0:
        print("\nCheck job status with: squeue -u $USER")
        print("Or use: python submit_and_postprocess.py --check-queue")
        print(f"When complete, run: python submit_and_postprocess.py --mode postprocess")


if __name__ == "__main__":
    main()
