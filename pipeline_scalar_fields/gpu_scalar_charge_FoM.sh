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

#SBATCH --exclude=sonicgpu1,sonicgpu2,sonicgpu3,sonicgpu4,sonicgpu5,sonicgpu6,sonicgpu7,sonicgpu8,sonicgpu9,sonicgpu10,sonicgpu11,sonicgpu12,sonicgpu13,sonicgpu14

# specify the walltime e.g 2 mins
#SBATCH -t 03:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

#SBATCH --output=output_%j.txt

# run from current directory
cd $SLURM_SUBMIT_DIR

# command to use
nvidia-smi
nvidia-smi -q | grep "Compute Capability"

#conda activate few_kerr2 #in few_kerr2 c'è installato few di soton , lisa-on-gpu senza EqualArms e FoM con tue modifiche
conda activate few_v2
module load cuda/12.6
which nvcc

### MASSIVE CASE ###
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 5 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.018  --repo scalar_results/results_FoM/massive --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 8 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.018  --repo scalar_results/results_FoM/massive --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.018  --repo scalar_results/results_FoM/massive --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 5 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.036  --repo scalar_results/results_FoM/massive --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 8 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.036  --repo scalar_results/results_FoM/massive --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 1.5 --z 0.1 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.01 --ScalarMass 0.036  --repo scalar_results/results_FoM/massive --calculate_fisher 1

###MASSLESS CASE Lambda=d^2 != 0 ###
python pipeline_scalar_charge_FoM.py  --M 5e5 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0025 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 5e5 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.000625 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 5e5 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0004 --ScalarMass 0.0 --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 5e5 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.000225 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 5e5 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0001 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1

python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0025 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.000625 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0004 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.000225 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0001 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.000049 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1


#UPPER BOUNDS MASSLESS CASE : Injection with Lambda= d^2 = 0
python pipeline_scalar_charge_FoM.py  --M 5e5 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1
python pipeline_scalar_charge_FoM.py  --M 1e6 --mu 10 --a 0.9 --e_f 0.0 --T 2.0 --z 0.2 --psd_file ../pipeline/TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --use_scalar_charge --Lambda 0.0 --ScalarMass 0.0  --repo scalar_results/results_FoM/massless --calculate_fisher 1

conda deactivate