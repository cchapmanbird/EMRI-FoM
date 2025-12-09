#!/usr/bin/env python3
"""
Submit postprocessing jobs to SLURM for SNR folders.

Usage:
    # Submit postprocessing jobs for snr_0 to snr_57
    python submit_and_postprocess.py --num-jobs 58
    
    # Check current queue status
    python submit_and_postprocess.py --check-queue
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def submit_postprocess_job(source_folder, job_num):
    """
    Submit a SLURM job for postprocessing a single SNR folder.
    
    Args:
        source_folder (str): Path to SNR folder (e.g., "snr_0/")
        job_num (int): Job number for naming
    """
    job_script = f"slurm_postprocess_{job_num}.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH -p normal
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -e postprocess_{job_num}.err
#SBATCH -o postprocess_{job_num}.out
#SBATCH --job-name=EMRI_postprocess_{job_num}
#SBATCH -t 12:00:00

cd $HOME/GitHub/EMRI-FoM/pipeline/

# Run postprocessing for this specific folder
python postprocess_snr.py {source_folder}

echo "Postprocess job {job_num} for {source_folder} completed"
"""
    
    with open(job_script, 'w') as f:
        f.write(script_content)
    
    os.chmod(job_script, 0o755)
    
    # Submit job
    try:
        result = subprocess.run(["sbatch", job_script], capture_output=True, text=True, check=True)
        job_id_slurm = result.stdout.strip().split()[-1]
        print(f"✓ Submitted job {job_num}: {job_id_slurm} for {source_folder}")
        os.remove(job_script)
        return job_id_slurm
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job {job_num}: {e}")
        print(f"  Error: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Submit postprocessing jobs to SLURM for SNR folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit postprocessing jobs for all 58 SNR folders
  python submit_and_postprocess.py --num-jobs 58
  
  # Check current queue status
  python submit_and_postprocess.py --check-queue
  
  # Submit for specific range (e.g., snr_0 to snr_10)
  python submit_and_postprocess.py --num-jobs 11
        """
    )
    
    parser.add_argument("--num-jobs", type=int, default=58,
                       help="Number of SNR folders to process (snr_0 to snr_N-1, default: 58)")
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
    
    # Submit postprocessing jobs
    print(f"Submitting {args.num_jobs} postprocessing jobs...")
    print(f"Processing folders: snr_0 to snr_{args.num_jobs - 1}\n")
    
    submitted = 0
    failed = 0
    job_ids = []
    
    for i in range(args.num_jobs):
        source_folder = f"snr_{i}/"
        job_id = submit_postprocess_job(source_folder, i)
        
        if job_id:
            submitted += 1
            job_ids.append(job_id)
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Submitted: {submitted} jobs")
    print(f"  Failed: {failed} jobs")
    print(f"{'='*60}\n")
    
    if job_ids:
        print("Monitor job status with:")
        print(f"  squeue -u $USER")
        print(f"\nOr check specific jobs with:")
        print(f"  squeue -j {','.join(job_ids[:5])}... (showing first 5)")


if __name__ == "__main__":
    main()