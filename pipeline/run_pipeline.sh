# pipeline execution for RedBook sources
python generate_source_backwards.py RedBook_Sources/EMRI_circular/EMRI_circular_sourceframe.npy example_detector_frame.npy --psd_file TDI2_AE_psd.npy --dt 10 --use_gpu --N_montecarlo 10 --seed --device 3
python fim_EMRI.py example_detector_frame.npy RedBook_Sources/EMRI_circular  --psd_file TDI2_AE_psd.npy --dt 10 --use_gpu --seed --device 3
