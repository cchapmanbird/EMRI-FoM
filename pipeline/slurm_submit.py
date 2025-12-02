#!/usr/bin/env python3
# filepath: slurm_submit.py
"""
Python script to submit SLURM jobs for EMRI FoM pipeline.
Usage: 
    python slurm_submit.py --mode snr [--test]
    python slurm_submit.py --mode pe [--test]
"""

import os
import sys
import subprocess
import json
import numpy as np
from pathlib import Path
import argparse
import json

def submit_slurm_job(source_params, pipeline_script="pipeline.py", partition="gpu_a100_7c"):
    """
    Submit a single SLURM job for an EMRI source.
    
    Args:
        source_params (dict): Dictionary containing source parameters
        pipeline_script (str): Path to the pipeline.py script
        partition (str): SLURM partition to use, default is 'gpu_a100_7c' alternatively 'gpu_a100_22c'
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
    python_cmd += f"    --model {source_params['model']} \\\n"
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

echo "Job completed successfully"
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
    
    Args:
        test_mode (bool): If True, only generate 1 source for testing
        repo_root (str): Prefix for repository names
        psd_file (str): PSD file to use
    
    Returns:
        list: List of source parameter dictionaries
    """
    # Parameters
    Nmonte = 1 if test_mode else 1000
    dev = 0
    channels = 'AET'
    model = 'scirdv1'
    include_foreground = True
    esaorbits = True
    tdi2 = True
    
    sources = []
    
    with open("emri_pe_sources.json", "r") as f:
        source_dict = json.load(f)
    
    # consider only a specific source for test mode
    # if test_mode:
    selected_source = 0

    for redshift in np.logspace(-3, 1, 10):
        for key, params in source_dict.items():
            # if int(key) != selected_source:
            #     continue
            
            m1 = params["m1"]
            m2 = params["m2"]
            a = params["a"]
            ef = params["e_f"]
            Tobs = params["Tpl"]
            dt = params["dt"]
            
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
                "model": model,
                "channels": channels,
                "dt": dt,
                "N_montecarlo": Nmonte,
                "device": dev,
                "pe": 0,
                "extra_args": extra_args.strip(),
            })

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
    
    Args:
        test_mode (bool): If True, only generate 1 source for testing
        repo_root (str): Prefix for repository names
        psd_file (str): PSD file to use
    
    Returns:
        list: List of source parameter dictionaries
    """
    # Parameters
    Nmonte = 1 if test_mode else 10
    dev = 0
    channels = 'AET'
    model = 'scirdv1'
    include_foreground = True
    esaorbits = True
    tdi2 = True
    
    sources = []
    
    with open("emri_pe_sources_with_z_ref.json", "r") as f:
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
                "model": model,
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


def check_queue():
    """Check SLURM queue status"""
    try:
        result = subprocess.run(["squeue", "-u", os.getenv("USER")], 
                              capture_output=True, text=True, check=True)
        print("Current queue status:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to check queue: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for EMRI FoM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit SNR calculation jobs (production mode)
  python slurm_submit.py --mode snr
  
  # Submit PE calculation jobs (test mode)
  python slurm_submit.py --mode pe --test
  
  # Check current queue status
  python slurm_submit.py --check-queue
  
  # Use different GPU partition
  python slurm_submit.py --mode snr --partition gpu_a100_22c
  
  # Use custom PSD file
  python slurm_submit.py --mode snr --psd TDI2_AE_psd_emri_background_1.5_yr.npy
  
  # Final production runs:
  python slurm_submit.py --mode pe --psd TDI2_AE_psd_emri_background_1.5_yr.npy --partition gpu_a100_22c
  python slurm_submit.py --mode snr --psd TDI2_AE_psd_emri_background_1.5_yr.npy --partition gpu_a100_22c
  
        """
    )
    
    parser.add_argument("--mode", choices=["snr", "pe"], 
                       help="Pipeline mode: 'snr' for SNR calculations, 'pe' for parameter estimation")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode (1 source, 1 Monte Carlo)")
    parser.add_argument("--check-queue", action="store_true", 
                       help="Check current SLURM queue status")
    parser.add_argument("--partition", type=str, default="gpu_a100_22c",
                       help="SLURM partition to use (default: gpu_a100_7c, optional: gpu_a100_22c)")
    parser.add_argument("--psd", type=str, 
                       choices=["TDI2_AE_psd_emri_background_1.5_yr.npy", "TDI2_AE_psd_emri_background_4.5_yr.npy"],
                       default="TDI2_AE_psd_emri_background_4.5_yr.npy",
                       help="PSD file to use (default: TDI2_AE_psd.npy)")
    
    args = parser.parse_args()
    
    # Check queue status if requested
    if args.check_queue:
        check_queue()
        return
    
    # Validate mode is specified
    if not args.mode:
        parser.error("--mode is required unless using --check-queue")
    
    # Change to pipeline directory
    pipeline_dir = Path(__file__).parent
    os.chdir(pipeline_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Generate sources based on mode
    if args.mode == "snr":
        repo_root = "test_snr_" if args.test else "production_snr_"
        sources = generate_snr_sources(test_mode=args.test, repo_root=repo_root, psd_file=args.psd)
    else:  # pe mode
        repo_root = "test_pe_" if args.test else "production_inference_"
        print("PSD set internally")
        sources = generate_pe_sources(test_mode=args.test, repo_root=repo_root)
    
    print(f"\nSubmitting {len(sources)} jobs in {args.mode} mode...")
    print(f"Partition: {args.partition}")
    print(f"PSD file: {args.psd}")
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
        print("Or use: python slurm_submit.py --check-queue")


if __name__ == "__main__":
    main()
