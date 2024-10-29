"""
Parameter estimation precision for EMRIs, averaged over nuisance parameters
"""
import logging
import argparse
import numpy as np
from common import standard_cosmology, get_covariance_matrix
from few.waveform import GenerateEMRIWaveform, EMRIInspiral
from few.utils.utility import get_p_at_t, get_separatrix
from fastlisaresponse import ResponseWrapper
from scipy.interpolate import CubicSpline

try:
    import cupy as xp
except:
    print("GPU import failed")
    xp = np

def draw_sources(fix_params, N_per_source):
    Nsources = fix_params.shape[0]
    params_out = np.zeros((Nsources, N_per_source, 14))

    # fill fixed parameters
    params_out[:,:,:7] = fix_params[:,None,:]

    # random draws of other parameters
    params_out[:,:,7] = np.pi/2 - np.arcsin(np.random.uniform(-1, 1, (Nsources, N_per_source)))
    params_out[:,:,8] = np.random.uniform(0, 2*np.pi, (Nsources, N_per_source))
    params_out[:,:,9] = np.pi/2 - np.arcsin(np.random.uniform(-1, 1, (Nsources, N_per_source)))
    params_out[:,:,10] = np.random.uniform(0, 2*np.pi, (Nsources, N_per_source))
    params_out[:,:,11] = np.random.uniform(0, 2*np.pi, (Nsources, N_per_source))
    params_out[:,:,12] = np.random.uniform(0, 2*np.pi, (Nsources, N_per_source))
    params_out[:,:,13] = np.random.uniform(0, 2*np.pi, (Nsources, N_per_source))

    return params_out

logger = logging.getLogger()
parser = argparse.ArgumentParser()

parser.add_argument("model", help="Waveform model to use")
parser.add_argument("psd_file", help="Path to a file containing PSD frequency-value pairs")
parser.add_argument("dt", help="Sampling cadence in seconds", type=float)
parser.add_argument("T", help="Waveform duration in years.", type=float)

parser.add_argument("--use_gpu", help="Whether to use GPU for FIM computation", action="store_true")
parser.add_argument("--parameter_file", help="Path to a file containing source parameters, organised as (N_source, N_montecarlo, N_params)")
parser.add_argument("--deltas_file", help="Path to a file containing stable delta values for all sources, organised as (N_source, N_montecarlo, N_params)")
parser.add_argument("--seed", help="numpy seed for random operations.",  action="store_const", const=42)
args = parser.parse_args()

np.random.seed(args.seed)

base_wave = GenerateEMRIWaveform(args.model, use_gpu=args.use_gpu)

# order of the langrangian interpolation
order = 25

orbit_file_esa = "../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"

orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "2nd generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
)

model = ResponseWrapper(
    base_wave,
    args.T,
    args.dt,
    index_lambda,
    index_beta,
    t0=10000.,
    flip_hx=True,  # set to True if waveform is h+ - ihx
    use_gpu=args.use_gpu,
    remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=False,  # False if using polar angle (theta)
    remove_garbage=True,  # removes the beginning of the signal that has bad information
    **tdi_kwargs_esa,
)

traj_model = base_wave.waveform_generator.inspiral_generator

if args.parameter_file is None:
    logger.critical("--parameter_file not supplied. This file will now be produced.")
    
    Ms = [1e6] * 5
    mu = [1e1, 3e1, 1e1, 1e1, 1e1]
    ecc = [0.6, 0.6, 0.6, 0.01, 0.6]
    spin = [0.89, 0.89, 0.5, 0.89, 0.1]
    xI = [1.] * 5

    redshift = 1.
    dists = [standard_cosmology(H0=67.).dl_zH0(redshift) / 1000.] * 5

    p0 = []
    print("getting p0 values")
    for i in range(len(Ms)):
        print(i)
        p0v = get_p_at_t(traj_model, args.T*1.001, [Ms[i], mu[i], spin[i], ecc[i], xI[i]], bounds=[get_separatrix(spin[i], ecc[i], xI[i]) + 3., 16.])
        p0.append(p0v)

    fixed_pars = np.vstack((Ms, mu, spin, p0, ecc, xI, dists)).T
    params_file = draw_sources(fixed_pars, N_per_source=10)

    np.save("EMRI_parameters", params_file)

else:
    params_file = np.load(args.parameter_file)

if args.deltas_file is None:
    logger.critical("--deltas_file not supplied. These will be manually obtained for each source and saved to a file at the end.")
    deltas_file=None

else:
    logger.info("Received stable deltas file. These step sizes will be used for FIM computation.")
    deltas_file = np.load(args.deltas_file)

    assert params_file.shape == deltas_file.shape

psdf, psdv = np.load(args.psd_file).T
psd_interp = CubicSpline(psdf, psdv)
psd_wrap = lambda f, **kwargs: psd_interp(f)

# run a test covariance matrix to check everything is working and determine shape
cov, fisher_object = get_covariance_matrix(
    model, 
    parameters=params_file[0,0], 
    dt=args.dt, 
    T=args.T, 
    psd=psd_wrap, 
    use_gpu=args.use_gpu,
    outdir="emri_temp", 
    deltas=deltas_file[0,0] if deltas_file is not None else None
)

print("Covariance is")
print(cov)

print("Deltas:")
print(fisher_object.deltas)

Nsources = params_file.shape[0]
N_montecarlo = params_file.shape[1]

cov_out = np.zeros((Nsources, N_montecarlo, cov.shape[0], cov.shape[0]))
deltas_cache = np.zeros_like(params_file)

for source_num in range(Nsources):
    for montecarlo_num in range(N_montecarlo):
        print(f"(Source num, random draw num) = ({Nsources, N_montecarlo})")
        if deltas_file is not None:
            deltas = deltas_file[source_num, montecarlo_num]
        else:
            deltas=None
        
        cov_here, fisher_object = get_covariance_matrix(
            model,
            parameters=params_file[source_num, montecarlo_num],
            dt=args.dt,
            T=args.T,
            psd=psd_wrap,
            use_gpu=args.use_gpu,
            outdir="emri_temp",
            deltas=deltas
        )

        cov_out[source_num, montecarlo_num] = cov_here

        if deltas is None:
            deltas_cache[source_num, montecarlo_num] = [fisher_object.deltas[nm] for nm in fisher_object.param_names]

print("Fisher matrices all computed")

np.save("EMRI_cov_montecarlo", cov_out)
if deltas is None:
    np.save("EMRI_deltas", deltas_cache)

print("Averaging covariance matrix diagonals to produce FoMs")

diagonals = np.array([cov_out[:,:,i,i] for i in range(cov_out.shape[2])])

# normalise to get relative precisions
diagonals /= params_file

diagonals_avg = np.mean(diagonals, axis=1)

np.save("EMRI_precisions", diagonals_avg)

#TODO: do we want to report a variance as well?