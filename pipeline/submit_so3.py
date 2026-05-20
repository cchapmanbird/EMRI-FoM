#!/usr/bin/env python3
"""
Submit SLURM jobs for EMRI SNR and parameter estimation calculations.

Usage:
    # Submit SNR jobs
    python submit_so3.py --mode snr [--test]
    
    # Submit PE jobs
    python submit_so3.py --mode pe [--test]
    
    # Check current queue status
    python submit_so3.py --check-queue
"""

import os
import sys
import subprocess
import json
import numpy as np
from pathlib import Path
import argparse
import glob


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
singularity exec --nv ../fom_resp {python_cmd}

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


def generate_snr_sources(test_mode=False, repo_root="snr_"):
    """
    Generate source parameters for SNR calculations.
    Based on pipeline_snr.py logic.
    """
    Nmonte = 1 if test_mode else 1000
    dev = 0
    channels = 'AE'
    include_foreground = True
    esaorbits = True
    tdi2 = True
    
    sources = []
    
    with open("so3_sources_Dec8.json", "r") as f:
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


def generate_pe_sources(test_mode=False, repo_root="inference_"):
    """
    Generate source parameters for parameter estimation (Fisher matrix).
    Based on pipeline_pe.py logic.
    """
    Nmonte = 1 if test_mode else 1000
    dev = 0
    channels = 'AE'
    include_foreground = True
    esaorbits = True
    tdi2 = True
    
    sources = []
    
    with open("so3_sources_Dec8.json", "r") as f:
        root_dict = json.load(f)
    
    for e_f in [0.0, 0.01]:
        for T_run in [0.25, 1.5, 4.5]:
            for key, params in root_dict.items():
                # read the redshift from the corresponding SNR inference file
                with open(f"./snr_{key}/inference_so3_sources_Dec8.json", "r") as f:
                    source_dict = json.load(f)
                    params = source_dict[key]

                z = params["z_ref_median"]
                Tobs = params["Tpl"]
                # make sure z is valid
                if (z == -1.0) or (Tobs > 0.5):
                    continue
                
                Tobs = T_run
                print(f"Generating PE source for {key} with e_f={e_f}, Tobs={Tobs}, z={z}")
                # run first the circular case, then the eccentric case
                if e_f != 0.0:
                    ef = e_f
                else:
                    ef = params["e_f"]

                m1 = params["m1"]
                m2 = params["m2"]
                a = params["a"]
                dt = params["dt"]
                
                
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


def generate_m1_sources(test_mode=False, repo_root="M1_inference_"):
    """
    Generate source parameters for parameter estimation (Fisher matrix)
    from the M1 catalog (M1_catalog.json).
    """
    Nmonte = 1 if test_mode else 1000
    dev = 0
    channels = 'AE'
    include_foreground = True
    esaorbits = True
    tdi2 = True

    sources = []

    with open("M1_catalog.json", "r") as f:
        catalog = json.load(f)

    for key, params in catalog.items():
        m1 = params["m1"]
        m2 = params["m2"]
        a = params["a"]
        z = params["redshift"]
        ef = params["e_f"]
        Tobs = params["Tpl"]
        dt = params["dt"]

        print(f"Generating M1 PE source for {key} with e_f={ef}, Tobs={Tobs}, z={z}")

        psd_file = "TDI2_AE_psd_emri_background_1.5_yr.npy"

        psd_name = psd_file.replace('.npy', '')
        source_name = (
            repo_root + key + '/'
            + f"m1={m1}_m2={m2}_a={a}_e_f={ef}_T={Tobs}_z={z}_{psd_name}"
        )

        # Build extra_args
        extra_args = ""
        if include_foreground:
            extra_args += " --foreground"
        if esaorbits:
            extra_args += " --esaorbits"
        if tdi2:
            extra_args += " --tdi2"

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
        sources = sources[-1:]

    # Save sources to file
    sources_file = repo_root + "sources_m1.txt"
    with open(sources_file, "w") as f:
        for source in sources:
            f.write(f"{source}\n")

    print(f"Generated {len(sources)} M1 PE sources")
    return sources


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for EMRI SNR and parameter estimation calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit SNR calculation jobs (production mode)
  python submit_so3.py --mode snr
  
  # Submit PE calculation jobs (test mode)
  python submit_so3.py --mode pe --test
  
  # Submit M1 catalog PE jobs
  python submit_so3.py --mode m1
  
  # Check current queue status
  python submit_so3.py --check-queue
        """
    )
    
    parser.add_argument("--mode", choices=["snr", "pe", "m1"], 
                       help="Pipeline mode: 'snr' for SNR calculations, 'pe' for parameter estimation, 'm1' for M1 catalog PE")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode (1 source, 1 Monte Carlo)")
    parser.add_argument("--partition", type=str, default="gpu_a100_22c",
                       help="SLURM partition to use for job submission (default: gpu_a100_22c)")
    parser.add_argument("--psd", type=str, 
                       choices=["TDI2_AE_psd_emri_background_1.5_yr.npy", "TDI2_AE_psd_emri_background_4.5_yr.npy"],
                       default="TDI2_AE_psd_emri_background_4.5_yr.npy",
                       help="PSD file to use")
    
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
    
    # Job submission modes
    if not args.mode:
        parser.error("--mode is required unless using --check-queue")
    
    # Generate sources based on mode
    if args.mode == "snr":
        repo_root = "test_snr_" if args.test else "snr_"
        sources = generate_snr_sources(test_mode=args.test, repo_root="./data/" + repo_root)
    elif args.mode == "pe":
        repo_root = "test_pe_" if args.test else "inference_"
        sources = generate_pe_sources(test_mode=args.test, repo_root="./data/" + repo_root)
    else:  # m1 mode
        repo_root = "test_M1_inference_" if args.test else "M1_inferenceNew1h_" #"M1_inference_"
        sources = generate_m1_sources(test_mode=args.test, repo_root="./data/" + repo_root)
    
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
        print("Or use: python submit_so3.py --check-queue")


if __name__ == "__main__":
    main()