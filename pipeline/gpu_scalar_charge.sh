#!/bin/bash -l
#SBATCH --job-name=hello
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=gpu
# Request 1 gpus
#SBATCH --gres=gpu:1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=1

#SBATCH --exclude=sonicgpu1,sonicgpu2,sonicgpu3,sonicgpu4,sonicgpu5,sonicgpu6,sonicgpu7,sonicgpu8

# specify the walltime e.g 2 mins
#SBATCH -t 25:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

#SBATCH --output=output_%j.txt

# run from current directory
cd $SLURM_SUBMIT_DIR

# command to use
nvidia-smi
nvidia-smi -q | grep "Compute Capability"


conda activate few_v2
module load cuda/12.6
which nvcc


#python pipeline_S05_FoM.py --M 1e6 --mu 1.4 --a 0.9 --e_f 0.0 --T 1.0 --z 0.08 --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.036  --repo test_acc --calculate_fisher 1
#python pipeline_S05_FoM.py --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.036  --repo scalar_results --calculate_fisher 1
python pipeline_scalar_charge.py  --M 1e6 --mu 100 --a 0.99 --e_f 0.0 --T 0.45 --z 0.5 --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1000 --device 0 --use_scalar_charge --Lambda 0.0 --ScalarMass 0.0  --repo scalar_results/massless_montecarlo/M1e6mu100a099z05dt10T45derorder8 --calculate_fisher 1
#python postprocess_scalar.py
#python flux_scalar_plot.py

conda deactivate