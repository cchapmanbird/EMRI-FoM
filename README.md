# EMRI Figures of Merit (FoMs) Computation

This repository contains codes for computing Figures of Merit (FoMs) related to Extreme Mass Ratio Inspirals (EMRIs).
## TODO

List of tasks:
- obtain Fisher for Red Book sources
- horizon redshift for sources
- define what good is

## Installation Instructions

Follow these steps to set up the environment and install the necessary packages:

**Install Python Packages and Fast EMRI Waveforms**

```sh
conda create -n fom_env python=3.12
conda activate fom_env
pip install multispline pygments matplotlib jupyter lisaanalysistools pandas Cython
export PATH=$PATH:/usr/local/cuda-12.5/bin/
pip install fastemriwaveforms-cuda12x --extra-index-url https://test.pypi.org/simple/
```

Test the installation device by running python
```python
import few
print(few.cutils.fast.__backend__)
```

**Install Stable EMRI Fisher Package**

```sh
cd StableEMRIFisher-package/
python -m pip install .
cd ..
```

**Install `lisa-on-gpu` for LISA Response** cloned from https://github.com/mikekatz04/lisa-on-gpu.git
```sh
cd local_response
python scripts/prebuild.py
python -m pip install .
cd ..
```

Verify `lisa-on-gpu` Installation by opening a Python shell and run:

```python
from fastlisaresponse import ResponseWrapper
```