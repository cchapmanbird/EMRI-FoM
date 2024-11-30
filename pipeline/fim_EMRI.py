"""
Parameter estimation precision for EMRIs, averaged over nuisance parameters
python fim_EMRI.py test example_psd.npy 10.0 1.0 --use_gpu --device 7 --deltas_file EMRI_deltas.npy 
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
try:
    import cupy as xp
except:
    print("GPU import failed")
    xp = np

logger = logging.getLogger()
parser = argparse.ArgumentParser()

# Add command line arguments
parser.add_argument("outdir", help="Where to save files")
parser.add_argument("psd_file", help="Path to a file containing PSD frequency-value pairs")
parser.add_argument("dt", help="Sampling cadence in seconds", type=float)
parser.add_argument("T", help="Waveform duration in years.", type=float)

parser.add_argument("--use_gpu", help="Whether to use GPU for FIM computation", action="store_true")
parser.add_argument("--parameter_file", help="Path to a file containing source parameters, organised as (N_source, N_montecarlo, N_params)")
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

# Generate EMRI waveform
base_wave = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", use_gpu=args.use_gpu, sum_kwargs=dict(pad_output=True))

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

# Initialize ResponseWrapper model
model = ResponseWrapper(
    base_wave,
    args.T,
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

# test model
# model(1e6, 10.0, 0.0, 10.0, 0.3, 1.0, 1.0, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3)

traj_model = base_wave.waveform_generator.inspiral_generator

# Generate or load parameter file
if args.parameter_file is None:
    logger.critical("--parameter_file not supplied. This file will now be produced.")
    
    Ms = [1e6, 5e5] 
    mu = [1e1, 1e1] 
    ecc = [0.3, 0.2] 
    spin = [0.9, 0.9] 
    xI = [1., 1.] 

    redshift = [1.0, 1.0]
    dists = [standard_cosmology(H0=67.).dl_zH0(el) / 1000. for el in redshift]

    p0 = []
    print("getting p0 values")
    for i in range(len(Ms)):
        p0v = get_p_at_t(traj_model, args.T*1.001, [Ms[i], mu[i], spin[i], ecc[i], xI[i]], 
                         bounds=[get_separatrix(spin[i], ecc[i], xI[i]) + 0.5, 16.]) # Bounds are very important here for working, ultimately we will just store the parameters
        p0.append(p0v)

    fixed_pars = np.vstack((Ms, mu, spin, p0, ecc, xI, dists)).T
    params_file = draw_sources(fixed_pars, N_per_source=1, seed=args.seed)

    np.save(args.outdir + "/EMRI_parameters", params_file)

else:
    params_file = np.load(args.parameter_file)

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

# Run a test covariance matrix to check everything is working and determine shape
cov, fisher_object = get_covariance_matrix(
    model, 
    parameters=params_file[0,0], 
    dt=args.dt, 
    T=args.T, 
    psd=psd_wrap, 
    use_gpu=args.use_gpu,
    outdir=args.outdir, 
    deltas=deltas_file[0,0] if deltas_file is not None else None
)

print("Covariance is")
print(cov)

print("Deltas:")
print(fisher_object.deltas)

Nsources = params_file.shape[0]
N_montecarlo = params_file.shape[1]

# Initialize output arrays
cov_out = np.zeros((Nsources, N_montecarlo, cov.shape[0], cov.shape[0]))
deltas_cache = np.zeros((Nsources, N_montecarlo, cov.shape[0]))

# Compute covariance matrices for all sources and Monte Carlo draws
for source_num in range(Nsources):
    for montecarlo_num in range(N_montecarlo):
        print("------------------------------------------------------------")
        print(f"(Source num, random draw num) = ({Nsources, N_montecarlo})")
        if deltas_file is not None:
            deltas = deltas_file[source_num, montecarlo_num]
        else:
            deltas=None
        
        # current folder
        current = '/source_{0}_draw_{1}'.format(source_num, montecarlo_num)
        cov_here, fisher_object = get_covariance_matrix(
            model,
            parameters=params_file[source_num, montecarlo_num],
            dt=args.dt,
            T=args.T,
            psd=psd_wrap,
            use_gpu=args.use_gpu,
            outdir=args.outdir + current,
            deltas=deltas,
            CovEllipse=True, 
            stability_plot=False,
        )

        # save fisher deltas
        fisher_object.save_deltas()
        # save covariance
        np.save(args.outdir + current + f"/source_{source_num}_draw_{montecarlo_num}_cov", cov_here)
        # save params
        breakpoint()
        np.save(args.outdir + current + f"/source_{source_num}_draw_{montecarlo_num}_params", params_file[source_num, montecarlo_num])
        # print relative precision
        print("Relative precision is")
        print(fisher_object.param_names)
        print(np.sqrt(np.diag(cov_here[0])) / np.delete(params_file[source_num, montecarlo_num], [5, 12]))

        cov_out[source_num, montecarlo_num] = cov_here

        if deltas is None:
            deltas_cache[source_num, montecarlo_num] = [fisher_object.deltas[nm] for nm in fisher_object.param_names]

print("Fisher matrices all computed")

# # Save covariance matrices and deltas
# np.save("EMRI_cov_montecarlo", cov_out)
# if deltas is None:
#     np.save("EMRI_deltas", deltas_cache)

# print("Averaging covariance matrix diagonals to produce FoMs")
# breakpoint()
# # Compute and save average precisions
# diagonals = np.array([cov_out[:,:,i,i] for i in range(cov_out.shape[2])])

# # Normalize to get relative precisions
# diagonals /= params_file

# diagonals_avg = np.mean(diagonals, axis=1)

# np.save(args.outdir + "/EMRI_precisions", diagonals_avg)

# TODO: do we want to report a variance as well?