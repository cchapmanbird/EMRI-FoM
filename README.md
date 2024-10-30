# EMRI Figures of Merit (FoMs) Computation

This repository contains codes for computing Figures of Merit (FoMs) related to Extreme Mass Ratio Inspirals (EMRIs).

## Installation Instructions

Follow these steps to set up the environment and install the necessary packages:

1. **Create a Virtual Environment**

    ```sh
    mamba create -n fom -c conda-forge -y gcc_linux-64 gxx_linux-64 h5py wget gsl liblapacke lapack openblas python=3.10
    mamba activate fom
    pip install numpy Cython scipy tqdm jupyter ipython requests rich matplotlib
    ```

2. **Clone the Repository**

    ```sh
    git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
    cd FastEMRIWaveforms
    git checkout Kerr_Equatorial_Eccentric
    ```

3. **Run the Prebuild Script**

    ```sh
    python scripts/prebuild.py
    ```

4. **Install CuPy and Set CUDA Path**

    ```sh
    export PATH=$PATH:/usr/local/cuda-12.5/bin/
    pip install cupy-cuda12x
    ```

5. **Install the Package**

    ```sh
    python setup.py install
    cd ..
    ```

6. **Verify FEW Installation**

    Open a Python shell and run:

    ```python
    from few.waveform import *
    ```

7. **Install `lisa-on-gpu` for LISA Response**

    ```sh
    cd lisa-on-gpu
    python setup.py install
    cd ..
    ```

8. **Verify `lisa-on-gpu` Installation**

    Open a Python shell and run:

    ```python
    from fastlisaresponse import ResponseWrapper
    ```

9. **Install LISA Analysis Tools**

    ```sh
    pip install lisaanalysistools
    ```

10. **Install Stable EMRI Fisher Package**

     ```sh
     cd StableEMRIFisher-package/
     python -m pip install .
     ```

11. **Run the Pipeline**

     ```sh
     cd pipeline
     python fim_EMRI.py FastKerrEccentricEquatorialFlux example_psd.npy 10. 3.5
     ```