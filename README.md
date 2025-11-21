# EMRI Figures of Merit (FoMs) Computation

This repository contains codes for computing Figures of Merit (FoMs) related to Extreme Mass Ratio Inspirals (EMRIs) and Intermediat Mass Ratio Inspirals (IMRIs). The codes contained in this repository are meant to be run on GPUs and a singularity image can be found [here](https://public.spider.surfsara.nl/project/lisa_nlddpc/emri_fom_container/).


![SNR Figure of Merit Example](pipeline/requirements_results/error_distribution_absolute_errors_a.png)
![PE Figure of Merit Example](pipeline/requirements_results/snr_redshift_requirement_allspins.png)

TODO:
- Create requirements for parameter estimation and upload everything in SO3 https://gitlab.in2p3.fr/LISA/lisa-fom/-/tree/develop?ref_type=heads
- Quadrupole moment https://arxiv.org/abs/gr-qc/0612029 , mapping in eq 43 of https://arxiv.org/pdf/gr-qc/0510129
- Check Fisher Information Stability and update with Shubham folder
- Update response to newest one (Maybe? we can also keep the old response)
- Check installation instructions

## Installation Instructions

Follow these steps to set up the environment and install the necessary packages. The installation is meant to be run on GPUs with CUDA compiler `nvcc`.

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment. **Note**: There is no available `conda` compiler for Windows. If you want to install for Windows, you will probably need to add libraries and include paths to the `setup.py` file.

### Fast EMRI Waveforms

Below is a quick set of instructions to install the Fast EMRI Waveform (FEW) package.

Create an environment for the figures of merit by installing the latest version of FEW:
```sh
conda create -n fom_env -c conda-forge -y --override-channels python=3.12 fastemriwaveforms-cuda12x
conda activate fom_env
pip install tabulate markdown pypandoc scikit-learn healpy lisaanalysistools seaborn corner scipy tqdm jupyter ipython h5py requests matplotlib eryn Cython
```
Check which of the above packages are actually needed

Test the installation of FEW on GPU by running python
```python
import few
few.get_backend("cuda12x")
```

### Fisher Information package

Install the Fisher information package
```sh
cd StableEMRIFisher-package/
pip install .
cd ..
```

### Install `lisa-on-gpu` for LISA Response
To install the response on GPUs, you need to locate where the `nvcc` compiler is and add it to the path. For instance it could be located in `/usr/local/cuda-12.5/bin/` and I would add it to the path with
```sh
export PATH=$PATH:/usr/local/cuda-12.5/bin/
```

Then I can install the response
```sh
pip install 
cd lisa-on-gpu
python setup.py install
cd ..
```

Verify `lisa-on-gpu` Installation by opening a Python shell and run:
```python
from fastlisaresponse import ResponseWrapper
```

### Test installation

Test the waveform and response in the main folder
```
python -m unittest test_waveform_and_response.py 
```

Test the pipeline
```
cd pipeline
python pipeline.py --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 4.0 --z 0.5 --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --power_law --repo test_acc --calculate_fisher 1
```

### Instructions for container on Spider

Connect to GPU partition ```srun -p gpu_a100_22c --pty bash -i -l``` or ```srun -p gpu_a100_7c --gpus=a100:1 --pty bash -i -l```. 

#### Container Construction
Build a container using `singularity build --nv --fakeroot fom_final.sif fom.def` or an editable container with:
```
singularity build --sandbox --nv --fakeroot fom fom.def
```

To edit an editable container `fom` open a shell:
```
singularity shell --writable --nv --fakeroot fom
```

Then you can install your favorite packages:
```
python3 -m pip install --upgrade pip
python -m pip install --no-cache-dir nvidia-cuda-runtime-cu12 astropy eryn fastemriwaveforms-cuda12x multiprocess optax matplotlib scipy jupyter interpax numba Cython lisaanalysistools tabulate scienceplots
python3 -c "import few; few.get_backend('cuda12x'); print('FEW installation successful')"

# Set compilers explicitly and unset conda variables
unset CC CXX CUDACXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDACXX=/usr/local/cuda/bin/nvcc
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++'

# install lisa on gpu and StableEMRIFisher-package
git clone https://github.com/cchapmanbird/EMRI-FoM.git emri_fom_temp
cd emri_fom_temp/lisa-on-gpu/
python3 setup.py install
cd ../StableEMRIFisher-package/
python -m pip install .
cd ..
python3 -m unittest test_waveform_and_response.py
```

Convert to editable container `fom` into a final image you can run
```
singularity build fom_final.sif fom
```

Test final image using
```
singularity exec --nv fom_final.sif python -m unittest test_waveform_and_response.py
```
An already built image can be found [here](https://public.spider.surfsara.nl/project/lisa_nlddpc/emri_fom_container/).

Use the final image
```
cd pipeline
singularity exec --nv ../fom_final.sif python pipeline.py --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 4.0 --z 0.5 --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --power_law --repo test_acc --calculate_fisher 1
```