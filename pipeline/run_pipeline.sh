# pipeline execution for RedBook sources
# 
N_MONTECARLO=1
DEVICE=3
DT=10

python generate_source_backwards.py RedBook_Sources/EMRI_circular/EMRI_circular_sourceframe.npy RedBook_Sources/EMRI_circular/detector_frame.npy --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --N_montecarlo $N_MONTECARLO --device $DEVICE
python fim_EMRI.py RedBook_Sources/EMRI_circular/detector_frame.npy RedBook_Sources/EMRI_circular --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --device $DEVICE
echo "EMRI_circular done"

python generate_source_backwards.py RedBook_Sources/LightIMRI/LightIMRI_sourceframe.npy RedBook_Sources/LightIMRI/detector_frame.npy --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --N_montecarlo $N_MONTECARLO --device $DEVICE
python fim_EMRI.py RedBook_Sources/LightIMRI/detector_frame.npy RedBook_Sources/LightIMRI --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --device $DEVICE
echo "LightIMRI done"

python generate_source_backwards.py RedBook_Sources/EMRI_ef01/EMRI_ef01_sourceframe.npy RedBook_Sources/EMRI_ef01/detector_frame.npy --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --N_montecarlo $N_MONTECARLO --device $DEVICE
python fim_EMRI.py RedBook_Sources/EMRI_ef01/detector_frame.npy RedBook_Sources/EMRI_ef01 --psd_file TDI2_AE_psd.npy --dt $DT --use_gpu --device $DEVICE
echo "EMRI_ef01 done"