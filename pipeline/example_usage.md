# Example how to run the pipeline

## Step 1. Create detector frame parameters at beginning of observation from source frame parameters at plunge

Have a .npy file containing source frame parameters at plunge organised as (N_source, N_params_source)

N_source: types of sources

N_params_source: right now should be organized as [M_s, mu_s, a, e_f, Y_f, z, Tobs]

Run: 

python generate_source_backwards.py example_source_frame.npy example_detector_frame.npy --psd_file TDI2_AE_psd.npy --dt 10 --use_gpu --N_montecarlo 10 --seed

This creates a example_detector_frame.npy with detector frame parameters for the different random sky localization,s for each source that have at least SNR = 25 (you can change this treshold by redifining SNR_treshold in line 203)

example_detector_frame.npy has the structure (N_source, N_draws, N_params) where 

N_draws=N_montecarlo: number of random sky localizations

N_params: right now organized as [M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, z, Tobs, SNR]

Note: for 'circular orbits' it is better to use e_f = 1e-5 - 5e-5 to guarantee the likelihood remains approximately Gaussian

## Step 2. Run the Fisher for the different sources 

Run: 

python fim_EMRI.py example_detector_frame.npy example_outdir --psd_file TDI2_AE_psd.npy -dt 10 --use_gpu --seed 

This runs the fisher matrix for all the sources provided in example_detector_frame.npy and saves them in the example_outdir directory. You can also specify --Tobs, if you do not it automatically uses the Tobs saved in example_detector_frame.npy file


## Step 3. Analyze source types and get the average uncertanties (or take a specific draw from a source type)

Run:

python postprocess_fim.py example_oudtir --src_numb 0 

You can also specificy --src_draw to get just the uncertanties for a specific sky localization