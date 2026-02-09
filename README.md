# LISA EMRI/IMRI Figures of Merit

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/lorenzsp/emri-imri-fom)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzsp/EMRI-FoM/main?filepath=pipeline/degradation_analysis.ipynb)

## Overview

This repository provides comprehensive tools for computing **Figures of Merit (FoMs)** for Extreme Mass Ratio Inspirals (EMRIs) and Intermediate Mass Ratio Inspirals (IMRIs) observable by the [Laser Interferometer Space Antenna (LISA)](https://www.lisamission.org/) mission.

### What are EMRIs and IMRIs?

EMRIs occur when stellar-mass compact objects (1-100 M☉) spiral into supermassive black holes (10⁴-10⁷ M☉), providing unique laboratories for studying the galactic centers and testing general relativity in the strong-field regime. IMRIs involve intermediate-mass black holes and offer insights into black hole formation and growth across cosmic time.

### What This Repository Does

We quantify LISA's capability to:
- **Detect** EMRIs and IMRIs across different mass ranges and redshifts
- **Characterize** source parameters (masses, spins, eccentricity, distance) with precision
- **Assess impact** of detector sensitivity degradation and mission duration on science objectives

### Interactive Exploration

**Try it yourself!** Explore how LISA sensitivity changes affect EMRI/IMRI detection and parameter estimation using our [**interactive notebook**](https://mybinder.org/v2/gh/lorenzsp/EMRI-FoM/main?filepath=pipeline/degradation_analysis.ipynb). No installation required - just click the Binder badge above to launch in your browser.

**Note:** Binder environments are pre-built using GitHub Actions, so launch time is typically under 1 minute. If you see a longer build time, the container may be updating with recent changes.

### Using This Work

**If you use any material from this repository** (code, data, figures, or results) in your research, please cite our work. See the [Citation](#citation) section below for details.

---

## Key Results

Our analysis provides quantitative insights into LISA's EMRI/IMRI science capabilities across the full parameter space:

<div align="center">

### Parameter Space Coverage
![Figure of Merit EMRI/IMRIs](pipeline/figures/emri_imri_masses_m1_m2.png)
*EMRI and IMRI parameter space: primary vs secondary mass configurations*

### Detection Horizon
![Horizon Redshift EMRI/IMRIs](pipeline/figures/z_at_snr.png)
*Maximum detection redshift as a function of system parameters*

### Signal-to-Noise Ratio
![FoM SNR EMRI/IMRIs](pipeline/figures/snr_fom_ranges_m2_1_Tpl_0.25_prograde_retrograde.png)
*SNR distribution for different orbital configurations (prograde vs retrograde)*

### Parameter Estimation Precision
![FoM Precision EMRI/IMRIs](pipeline/figures/scatter_relative_precision_a_m1_vs_m2_spin_a0.99.png)
*Spin measurement precision across the mass parameter space*

</div>

---

## Getting Started

### Quick Start (No Installation)

The easiest way to explore the analysis is through our [interactive Jupyter notebook](https://mybinder.org/v2/gh/lorenzsp/EMRI-FoM/main?filepath=pipeline/degradation_analysis.ipynb).

### Local Installation

To reproduce the full analysis pipeline on your own GPU-enabled system, follow the installation instructions below. The pipeline uses GPU acceleration for efficient waveform generation and Fisher matrix calculations.

**Pre-built Container**: A ready-to-use Singularity container is available [here](https://public.spider.surfsara.nl/project/lisa_nlddpc/emri_fom_container/).

---

## Installation with Conda

### Prerequisites

- CUDA-capable GPU with compute capability ≥ 7.0
- CUDA Toolkit 12.x with `nvcc` compiler
- [Anaconda](https://docs.anaconda.com/anaconda/install/) or Miniconda
- Linux or macOS (Windows not currently supported due to compiler limitations)

### Step 1: Create Environment and Install FEW

[Fast EMRI Waveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) provides GPU-accelerated waveform generation:

```bash
# Create environment with FEW
conda create -n fom_env -c conda-forge -y --override-channels \
    python=3.12 fastemriwaveforms-cuda12x

# Activate environment
conda activate fom_env

# Install additional dependencies
pip install tabulate markdown pypandoc scikit-learn healpy \
    lisaanalysistools seaborn corner scipy tqdm jupyter \
    ipython h5py requests matplotlib eryn Cython
```

**Test FEW installation:**
```python
import few
few.get_backend("cuda12x")  # Should complete without errors
```

### Step 2: Install Fisher Information Package

```bash
cd StableEMRIFisher-package/
pip install .
cd ..
```

### Step 3: Install LISA Response (lisa-on-gpu)

First, locate your CUDA compiler and add it to PATH:

```bash
# Find nvcc location (typically in /usr/local/cuda-*/bin/)
export PATH=$PATH:/usr/local/cuda-12.5/bin/
```

Then install the response package:

```bash
cd lisa-on-gpu
python setup.py install
cd ..
```

**Verify installation:**
```python
from fastlisaresponse import ResponseWrapper  # Should import without errors
```

### Step 4: Test Installation

Run the test suite:

```bash
# Test waveform and response
python -m unittest test_waveform_and_response.py

# Test pipeline with minimal configuration
cd pipeline
python pipeline.py --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 4.0 --z 0.5 \
    --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 \
    --device 0 --power_law --repo test_acc --calculate_fisher 1
```

If all tests pass, your installation is complete! ✓

---

## Container Installation (HPC Environments)

For high-performance computing clusters, we provide Singularity container instructions. These are particularly useful for reproducibility and deployment on shared systems.

### Quick Start with Pre-built Container

Download and use the ready-made container:

```bash
# Download container (if needed)
# Available at: https://public.spider.surfsara.nl/project/lisa_nlddpc/emri_fom_container/

# Test the container
singularity exec --nv fom_final.sif python -m unittest test_waveform_and_response.py

# Run pipeline analysis
cd pipeline
singularity exec --nv ../fom_final.sif python pipeline.py \
    --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 4.0 --z 0.5 \
    --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu \
    --N_montecarlo 1 --device 0 --power_law --repo test_acc \
    --calculate_fisher 1
```

### Building Your Own Container

#### Request GPU Node

Example for [Spider HPC](https://doc.spider.surfsara.nl/):
```bash
# Request GPU partition
srun -p gpu_a100_22c --mem 64G -G a100:1 -c 2 --pty bash
```

#### Build Container

Create a container from the definition file:

```bash
# Build final container
singularity build --nv --fakeroot fom_final.sif fom.def

# OR build editable sandbox for development
singularity build --sandbox --nv --fakeroot fom fom.def
```

#### Customize Editable Container

Open a shell in the editable container:

```bash
singularity shell --writable --nv --fakeroot fom
```

Install additional packages inside the container:

```bash
# Update pip
python -m pip install --upgrade pip

# Install core dependencies
python -m pip install --no-cache-dir nvidia-cuda-runtime-cu12 astropy \
    eryn fastemriwaveforms-cuda12x multiprocess optax matplotlib scipy \
    jupyter interpax numba Cython lisaanalysistools tabulate scienceplots \
    healpy pandas filelock

# Test FEW installation
python -c "import few; few.get_backend('cuda12x'); print('FEW installation successful')"

# Set compilers for building GPU packages
unset CC CXX CUDACXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDACXX=/usr/local/cuda/bin/nvcc
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++'

# Clone and install EMRI-FoM packages
git clone https://github.com/cchapmanbird/EMRI-FoM.git emri_fom_temp
cd emri_fom_temp/lisa-on-gpu/
python setup.py install
cd ../StableEMRIFisher-package/
python -m pip install .
cd ..

# Run tests
python -m unittest test_waveform_and_response.py
```

Convert sandbox to final image:

```bash
singularity build fom_final.sif fom
```

---

## Alternative: Python Virtual Environment

For systems without Singularity or conda, use a standard Python virtual environment:

```bash
# Request compute node (HPC example)
srun --partition=short --time=12:00:00 --pty bash -i -l

# Create and activate virtual environment
python -m venv fom_venv/
source fom_venv/bin/activate

# Install packages
python -m pip install --upgrade pip
python -m pip install --no-cache-dir nvidia-cuda-runtime-cu12 astropy \
    eryn fastemriwaveforms-cuda12x multiprocess optax matplotlib scipy \
    jupyter interpax numba Cython lisaanalysistools tabulate scienceplots \
    healpy pandas filelock

# Launch Jupyter (for remote work)
jupyter lab --ip="*" --no-browser

# On local machine, create SSH tunnel:
# ssh -NL 8888:wn-la-01:8888 spider
```

---

## Binder Performance Optimization

This repository uses **pre-built Docker containers** to significantly reduce Binder launch time:

- **GitHub Actions** automatically builds and pushes a Docker image whenever `requirements.txt` or key files change
- **Binder** pulls the pre-built image from GitHub Container Registry instead of building from scratch
- **Result**: Launch time reduced from 5-10 minutes to typically under 1 minute

The workflow is defined in [`.github/workflows/binder-build.yml`](.github/workflows/binder-build.yml) and uses [`repo2docker`](https://repo2docker.readthedocs.io/) to create the same environment that Binder would build.

### For Maintainers

After updating `requirements.txt`:
1. Commit and push changes to the `main` branch
2. GitHub Actions will automatically rebuild and push the Docker image
3. Subsequent Binder launches will use the updated pre-built image

To manually trigger a rebuild, go to the [Actions tab](../../actions/workflows/binder-build.yml) and click "Run workflow".

---

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{speri2025fom,
  title = {Figures of Merit for Extreme and Intermediate Mass Ratio Inspirals for the Laser Interferometer Space Antenna},
  author = {Speri, Lorenzo and Duque, Francisco and Santini, Alessandro and Kejriwal, Shubham and Chapman-Bird, Christian and Burke, Ollie and Buscicchio, Riccardo and Mangiagli, Alberto},
  journal = {In preparation},
  year = {2025},
  url = {https://github.com/lorenzsp/EMRI-FoM}
}
```

**Paper in preparation**: Speri, L., Duque, F., Santini, A., Kejriwal, S., Chapman-Bird, C., Burke, O., Buscicchio, R., & Mangiagli, A. (2025). *Figures of Merit for Extreme and Intermediate Mass Ratio Inspirals for the Laser Interferometer Space Antenna*. In preparation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

*Developed for the LISA mission - Opening new windows on the universe through gravitational waves*
