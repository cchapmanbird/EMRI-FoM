#!/usr/bin/env python3
"""
Submit postprocessing jobs to SLURM for SNR folders.

Usage:
    # Submit SNR postprocessing jobs for snr_0 to snr_57
    python submit_postprocess.py snr --num-jobs 58
    
    # Submit inference postprocessing jobs for snr_0 to snr_57
    python submit_postprocess.py inference --num-jobs 58
    
    # Check current queue status
    python submit_postprocess.py snr --check-queue
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def submit_job(script_name, job_type, source_folder, job_num):
    """
    Submit a SLURM job for postprocessing.
    
    Args:
        script_name (str): Name of script to run (e.g., "postprocess_snr.py")
        job_type (str): Type of job for naming (e.g., "snr", "inference")
        source_folder (str): Path to SNR folder (e.g., "snr_0/")
        job_num (int): Job number for naming
    """
    job_script = f"slurm_postprocess_{job_type}_{job_num}.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH -p normal
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -e postprocess_{job_type}_{job_num}.err
#SBATCH -o postprocess_{job_type}_{job_num}.out
#SBATCH --job-name=EMRI_postprocess_{job_type}_{job_num}
#SBATCH -t 12:00:00

cd $HOME/GitHub/EMRI-FoM/pipeline/
source fom_venv/bin/activate
python {script_name} {source_folder}
echo "Postprocess {job_type} job {job_num} for {source_folder} completed"
"""
    
    with open(job_script, 'w') as f:
        f.write(script_content)
    
    os.chmod(job_script, 0o755)
    
    try:
        result = subprocess.run(["sbatch", job_script], capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"✓ Submitted {job_type} job {job_num}: {job_id} for {source_folder}")
        os.remove(job_script)
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit {job_type} job {job_num}: {e}")
        return None


def main():
    """Submit postprocessing jobs to SLURM."""
    parser = argparse.ArgumentParser(
        description="Submit postprocessing jobs to SLURM for SNR folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python submit_postprocess.py snr --num-jobs 58
  python submit_postprocess.py inference --num-jobs 58
  python submit_postprocess.py snr --check-queue
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Postprocessing type')
    
    for job_type, script_name in [('snr', 'postprocess_snr.py'), 
                                   ('inference', 'postprocess_inference.py')]:
        subparser = subparsers.add_parser(job_type, help=f'Submit {job_type} postprocessing jobs')
        subparser.add_argument("--num-jobs", type=int, default=58,
                              help="Number of SNR folders to process (default: 58)")
        subparser.add_argument("--check-queue", action="store_true",
                              help="Check current SLURM queue status")
        subparser.set_defaults(script_name=script_name, job_type=job_type)
    
    args = parser.parse_args()
    
    pipeline_dir = Path(__file__).parent
    os.chdir(pipeline_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    if args.check_queue:
        try:
            result = subprocess.run(["squeue", "-u", os.getenv("USER")], 
                          capture_output=True, text=True, check=True)
            print("Current queue status:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to check queue: {e}")
        return
    
    print(f"Submitting {args.num_jobs} {args.job_type} postprocessing jobs...")
    print(f"Processing folders: {args.job_type}_0 to {args.job_type}_{args.num_jobs - 1}\n")
    
    job_ids = []
    for i in range(args.num_jobs):
        job_id = submit_job(args.script_name, args.job_type, f"{args.job_type}_{i}/", i)
        if job_id:
            job_ids.append(job_id)
    
    print(f"\n{'='*60}")
    print(f"Summary: Submitted {len(job_ids)} jobs, Failed: {args.num_jobs - len(job_ids)}")
    print(f"{'='*60}\n")
    
    if job_ids:
        print("Monitor with: squeue -u $USER")
        print(f"Check jobs: squeue -j {','.join(job_ids[:5])}...")



if __name__ == "__main__":
    main()