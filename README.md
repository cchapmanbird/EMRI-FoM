# EMRI Figures of Merit (FoMs) Computation

This repository contains codes for computing Figures of Merit (FoMs) related to Extreme Mass Ratio Inspirals (EMRIs) and Intermedia Mass Ratio Inspirals (IMRIs).

## Installation Instructions

Follow these steps to set up the environment and install the necessary packages. The installation is meant to be run on GPUs with CUDA compiler `nvcc`.

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment. **Note**: There is no available `conda` compiler for Windows. If you want to install for Windows, you will probably need to add libraries and include paths to the `setup.py` file.

### Fast EMRI Waveforms

Below is a quick set of instructions to install the Fast EMRI Waveform (FEW) package.

Create an environment for the figures of merit
```sh
conda create -n fom -c conda-forge gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.9 pandas fortran-compiler
conda activate fom
```

Locate where the `nvcc` compile is located and add it to the path, in my case it is located in `/usr/local/cuda-12.5/bin/`
```
export PATH=$PATH:/usr/local/cuda-12.5/bin/
```

Check the version of your compiler by running `nvcc --version` and install the corresponding FEW cuda version for running on GPUs:
```
pip install --pre fastemriwaveforms-cuda12x
pip install tabulate
pip install markdown pypandoc
```

Test the installation device by running python
```python
import few
few.get_backend("cuda12x")
```

### LISA Data Challenge and FoM packages

Install the `ldc` and the `fom` packages
```sh
git clone -b develop https://gitlab.in2p3.fr/LISA/LDC.git
cd LDC
pip install .
cd ..

git clone -b develop https://gitlab.in2p3.fr/lisa/lisa-fom.git
cd lisa-fom
pip install -r requirements.txt
pip install pandas==2.2.3
python setup.py install
cd ..
```

### Fisher Information package

Install the Fisher information package
```sh
cd StableEMRIFisher-package/
pip install .
cd ..
```

### Install `lisa-on-gpu` for LISA Response
Install the response
```sh
git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu
pip install lisaanalysistools
python scripts/prebuild.py
pip install .
cd ..
```

Verify `lisa-on-gpu` Installation by opening a Python shell and run:

```python
from fastlisaresponse import ResponseWrapper
```

### Test waveform and response
```
python -m unittest test_waveform_and_response.py 
```

## Running the pipeline

Place yourself in the folder pipeline and run the pipeline on GPUs
```
cd pipeline
python run_pipeline.py
```

Inside the `run_pipeline.py` you can modify the thresholds for the science objectives, the sources and the number of randomization over sky and phase.

You can also run the pipeline in the background with `nohup python run_pipeline.py > logpipeline.log &`