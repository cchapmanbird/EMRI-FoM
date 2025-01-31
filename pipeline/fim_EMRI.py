"""
Parameter estimation precision for EMRIs, averaged over nuisance parameters
python fim_EMRI.py science_obj example_psd.npy 10.0 --use_gpu --device 7 --parameter_file EMRI_parameters.npy 
"""
import os
import logging
import argparse
import numpy as np
from common import standard_cosmology, get_covariance_matrix, draw_sources
from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_p_at_t, get_separatrix
from fastlisaresponse import ResponseWrapper
from scipy.interpolate import CubicSpline
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot
try:
    import cupy as xp
except:
    print("GPU import failed")
    xp = np

logger = logging.getLogger()
parser = argparse.ArgumentParser()

# Add command line arguments
parser.add_argument("parameter_file", help="Path to a file containing source parameters, organised as (N_source, N_montecarlo, N_params)")
parser.add_argument("outdir", help="Where to save files")

parser.add_argument("--psd_file", help="Path to a file containing PSD frequency-value pairs", default = "TDI2_AE_psd.npy")
parser.add_argument("--dt", help="Sampling cadence in seconds", type=float, default = 10)
parser.add_argument("--Tobs", help="Waveform duration in years.", type=float)
parser.add_argument("--use_gpu", help="Whether to use GPU for FIM computation", action="store_true")
parser.add_argument("--deltas_file", help="Path to a file containing stable delta values for all sources, organised as (N_source, N_montecarlo, N_params)")
parser.add_argument("--seed", help="numpy seed for random operations.",  action="store_const", const=42)
parser.add_argument("--device", help="GPU device", type=int, default=0)

args = parser.parse_args()

# ceate folder with name args.outdir
os.makedirs(args.outdir, exist_ok=True)

# Set device for GPU
if args.use_gpu:
    print("Using GPU", args.device)
    xp.cuda.Device(args.device).use()
    xp.random.seed(args.seed)

np.random.seed(args.seed)

inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e5),
        "err": 1e-12,  
        #"use_rk4": True,
}

# Generate EMRI waveform
base_wave = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", inspiral_kwargs = inspiral_kwargs,  use_gpu=args.use_gpu, sum_kwargs=dict(pad_output=True))

#transformed wave
class transf_log_e_wave():
    def __init__(self, base_wave):
        self.base_wave = base_wave

    def __call__(self, *args, **kwargs):
        args = list(args)
        args[4] = np.exp(args[4]) #index of eccentricity on the FEW waveform call
        return self.base_wave(*args, **kwargs)
    
    def __getattr__(self, name):
        # Forward attribute access to base_wave
        return getattr(self.base_wave, name)

transf_log_e_wave = transf_log_e_wave(base_wave)

# Order of the langrangian interpolation
order = 25

# Orbit file and kwargs
orbit_file_esa = "../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

# TDI generation
tdi_gen = "2nd generation"
index_lambda = 8
index_beta = 7

# TDI kwargs
tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
)

#Load detector frame parameter file organized as (N_source, N_montecarlo, N_params), 
# where N_params are organized as [M, mu, a, p0, e0, Y0, dist, qS, phiS, thetaS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T_coalescence, SNR to coalescence]
params_file = np.load(args.parameter_file)

#If Tobs is set by the user, overwrite the Tobs in the parameter file
if args.Tobs is not None:
    params_file[:, :, -2] = args.T 

# Generate or load deltas file
if args.deltas_file is None:
    logger.critical("--deltas_file not supplied. These will be manually obtained for each source and saved to a file at the end.")
    deltas_file=None

else:
    logger.info("Received stable deltas file. These step sizes will be used for FIM computation.")
    deltas_file = np.load(args.deltas_file)

# Load PSD file and create interpolant
psdf, psdv = np.load(args.psd_file).T
psd_interp = CubicSplineInterpolant(psdf, psdv, use_gpu=args.use_gpu)
psd_wrap = lambda f, **kwargs: psd_interp(f)

# Initialize ResponseWrapper model and run a test covariance matrix to check everything is working and determine shape
model = ResponseWrapper(
        transf_log_e_wave,
        params_file[0, 0, -2],
        args.dt,
        index_lambda,
        index_beta,
        t0=100000.,
        flip_hx=True,  # Set to True if waveform is h+ - ihx
        use_gpu=args.use_gpu,
        remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=True,  # Removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )

# Run a test covariance matrix to check everything is working and determine shape
cov, fisher_object, SNR = get_covariance_matrix(
    model, 
    parameters=params_file[0,0,:-3], # Remove z, Tobs and SNR
    dt=args.dt, 
    T=params_file[0,0,-2], 
    psd=psd_wrap, 
    use_gpu=args.use_gpu,
    outdir=args.outdir, 
    log_e = True,
    deltas=deltas_file[0,0] if deltas_file is not None else None
)

print("Precision is")
print(np.sqrt(np.diag(cov)))


Nsources = params_file.shape[0]
N_montecarlo = params_file.shape[1]

# Initialize output arrays
cov_out = np.zeros((Nsources, N_montecarlo, cov.shape[0], cov.shape[0]))
deltas_cache = np.zeros((Nsources, N_montecarlo, cov.shape[0]))

# Compute covariance matrices for all sources and Monte Carlo draws
print(f"(Source num, random draw num) = ({Nsources, N_montecarlo})")
for source_num in range(Nsources):
    print("------------------------------------------------------------")
    
    # Initialize ResponseWrapper model for different sources (can have different T)
    model = ResponseWrapper(
        transf_log_e_wave,
        params_file[source_num, 0, -2], # Tobs
        args.dt,
        index_lambda,
        index_beta,
        t0=100000.,
        flip_hx=True,  # Set to True if waveform is h+ - ihx
        use_gpu=args.use_gpu,
        remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=True,  # Removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )

    
    for montecarlo_num in range(N_montecarlo):
        print("^^^^^^^^^^^^^^^^^^")
        print(f"working on (Source num, random draw num) = ({source_num, montecarlo_num})")
        if deltas_file is not None:
            deltas = deltas_file[source_num, montecarlo_num]
        else:
            deltas = None
        
        # current folder
        current = '/source_{0}_draw_{1}'.format(source_num, montecarlo_num)
        cov_here, fisher_object, SNR_here = get_covariance_matrix(
            model,
            parameters=params_file[source_num, montecarlo_num, :-3], # Remove z, Tobs and SNR
            dt=args.dt,
            T=params_file[source_num, montecarlo_num, -2],
            psd=psd_wrap,
            use_gpu=args.use_gpu,
            outdir=args.outdir + current,
            log_e = True,
            deltas=deltas,
        )

        params_file[source_num, montecarlo_num, -1] = SNR_here # Update SNR

        # save fisher deltas
        fisher_object.save_deltas()
        # save covariance
        np.save(args.outdir + current + f"/source_{source_num}_draw_{montecarlo_num}_cov", cov_here)
        # save params
        np.save(args.outdir + current + f"/source_{source_num}_draw_{montecarlo_num}_params", params_file[source_num, montecarlo_num])
        # print relative precision
        print("Precision is")
        print(fisher_object.param_names)
        print(np.sqrt(np.diag(cov_here))) #/ np.delete(params_file[source_num, montecarlo_num], [5, 12, 13, 14]))
        # save plot
        CovEllipsePlot(fisher_object.param_names, fisher_object.wave_params, cov_here, filename=args.outdir + current + f"/covariance_ellipses.png")
        cov_out[source_num, montecarlo_num] = cov_here

        if deltas is None:
            deltas_cache[source_num, montecarlo_num] = [fisher_object.deltas[nm] for nm in fisher_object.param_names]

print("Fisher matrices all computed") 