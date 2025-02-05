# pipeline execution for RedBook sources
# 
N_MONTECARLO=10
DEVICE=3
DT=5

python generate_source_backwards.py RedBook_Sources/EMRI_circular/EMRI_circular_sourceframe.npy RedBook_Sources/EMRI_circular/detector_frame.npy --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --N_montecarlo $N_MONTECARLO --seed --device $DEVICE
python fim_EMRI.py RedBook_Sources/EMRI_circular/detector_frame.npy RedBook_Sources/EMRI_circular  --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --seed --device $DEVICE

python generate_source_backwards.py RedBook_Sources/LightIMRI/LightIMRI_sourceframe.npy RedBook_Sources/LightIMRI/detector_frame.npy --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --N_montecarlo $N_MONTECARLO --seed --device $DEVICE
python fim_EMRI.py RedBook_Sources/LightIMRI/detector_frame.npy RedBook_Sources/LightIMRI  --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --seed --device $DEVICE

python generate_source_backwards.py RedBook_Sources/EMRI_ef01/EMRI_ef01_sourceframe.npy RedBook_Sources/EMRI_ef01/detector_frame.npy --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --N_montecarlo $N_MONTECARLO --seed --device $DEVICE
python fim_EMRI.py RedBook_Sources/EMRI_ef01/detector_frame.npy RedBook_Sources/EMRI_ef01  --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --seed --device $DEVICE
