import sys, os
import pickle as pkl
from time import time
import traceback
from copy import deepcopy
import pickle as pkl
import tracemalloc

import numpy as np
#from scipy.interpolate import InterpolatedUnivariateSpline as spline

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SchwarzEccFlux
from few.waveform.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux, FastSchwarzschildEccentricFluxBicubic

from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix

from fastlisaresponse import ResponseWrapper

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

from LISAfom.lisatools import lisa_parser, process_args, \
                              build_lisa_noise, build_lisa_orbits
from ldc.utils.logging import init_logger, close_logger
import lisaconstants as constants

import h5py

from scipy.signal import get_window
from tqdm import tqdm
import GPUtil

SEED = 26011996

WAVEFORM_ARGS = [FastSchwarzschildEccentricFluxBicubic]#[SchwarzschildEccentricWaveformBase, EMRIInspiral, RomanAmplitude, InterpolatedModeSum]
TRAJECTORY = SchwarzEccFlux
DEF_TOBS = 3.5
DEF_DT = 10.0
DEF_Z = 1
DEF_SNR_THR = 20

M_POINTS = 10.0**np.linspace(5.0,7.0,num=10)
MU_POINTS = [10.0, 30.0]   
Q_POINTS = [1e-5, 1e-4]

GRID_POINTS_Q = [[M, q * M]  for q in Q_POINTS for M in M_POINTS] #+ \
              #[[30., M] for M in np.logspace(5, np.log10(3e6), num=18)] + \
              #[[30., M] for M in np.logspace(np.log10(2e4), 5, num=18)]

GRID_POINTS_MU = [[M, mu] for mu in MU_POINTS for M in M_POINTS] #+ \

MU_MIN, MU_MAX = 1.0, 1e4

DEF_PARS = {
    'a': 0.0,
    'e0': 0.5,
    'x0': 1.0,
    'dist': 1,
    'qS': 1.141428995078945,
    'phiS': 1.8278083813267254,
    'qK': 1.5491394235138727,
    'phiK': 5.610551183647945,
    'Phi_phi0': 0.1,
    'Phi_theta0': 0.1,
    'Phi_r0': 0.1,
}

PARS_NAMES = [
    'M', 'mu', 'a', 'p0', 'e0', 'x0',
    'dist', 'qS', 'phiS', 'qK', 'phiK',
    'Phi_phi0', 'Phi_theta0', 'Phi_r0'
]



def get_free_gpus(n_gpus=1):
    '''
    Get the IDs of free GPUs.

    Parameters
    ----------
    n_gpus : int
        Number of free GPUs to return.
    
    Returns
    -------
    free_gpus : list
        List of IDs of free GPUs.
    '''

    free_gpus = GPUtil.getAvailable(order='first', limit=n_gpus, maxLoad=0.001, maxMemory=0.001)
    return free_gpus

try:
    import cupy as xp
    from cupyx.scipy.interpolate import Akima1DInterpolator as spline
    # set GPU device
    free_gpus = get_free_gpus(n_gpus=1)
    if not free_gpus:
        gpu_available = False
    else:
        os.system("CUDA_VISIBLE_DEVICES="+str(free_gpus[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
        print("Using GPU", free_gpus[0])
        gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    from scipy.interpolate import Akima1DInterpolator as spline
    gpu_available = False

import warnings
warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = False

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

np.random.seed(SEED)
xp.random.seed(SEED)

#------------------------------------------------------------

cosmo = Planck18 

def get_redshift(distance):
    return float(z_at_value(cosmo.luminosity_distance, distance * u.Gpc ))

def get_distance(redshift):
    return cosmo.luminosity_distance(redshift).to(u.Gpc).value

def to_cpu(x):
    try:
        return x.get()
    except AttributeError:
        return x

def compute_snr2(freqs, tdiA, tdiE, lisa_psd):
    """
    Compute the SNR of the waveform given the TDI channels and the LISA PSD
    """
    df = freqs[3] - freqs[2]
    
    return to_cpu(4.0 * df * xp.sum((xp.abs(tdiA)**2 + xp.abs(tdiE)**2)/lisa_psd(freqs)))


def get_p0(traj, pars, Tobs):
    try:    
        p0 = get_p_at_t(traj,Tobs * 0.999,[pars['M'], pars['mu'], pars['a'], pars['e0'], pars['x0']],bounds=[get_separatrix(pars['a'],pars['e0'],pars['x0'])+0.1, 40.0])
    except Exception as e:
        print(e)
        p0 = 21.0
    
    return p0

def randomize(pars, args):
    pars.update({
       'qS': np.arccos(np.random.uniform(-1, 1)),
       'phiS': np.random.uniform(0.0, 2 * np.pi),
       'qK': np.arccos(np.random.uniform(-1, 1)),
       'phiK': np.random.uniform(0.0, 2 * np.pi),
       'Phi_phi0': np.random.uniform(0.0, 2 * np.pi),
       'Phi_theta0': np.random.uniform(0.0, 2 * np.pi),
       'Phi_r0': np.random.uniform(0.0, 2 * np.pi)
    })

    if args.randomize_e:
        pars['e0'] = np.random.uniform(0.0, 0.7)

def get_def_pars(): #TODO look into this
    pars = DEF_PARS.copy()
    '''
    if _args.fixed_pars is None or \
        not os.path.exists(_args.fixed_pars):
        return _pars

    logger.info('Found the fixed parameters file %s.', _args.fixed_pars)

    with open(_args.fixed_pars, 'rb') as _fhand:
        fixed = pkl.load(_fhand)

    fixed_sel = [item for item in fixed if item['mu0'] == _mu]

    if len(fixed_sel) == 0:
        logger.info('Fixed parameters file has no entry for mu=%s', _mu)
        return _pars

    return dict(_pars, **fixed_sel[0])
    '''
    return pars

class wave_gen_windowed:
    """
    Generate a waveform and apply a window to it
    """
    def __init__(self, wave_gen, window_fn=('tukey', 0.005)):
        self.wave_gen = wave_gen
        self.window_fn = window_fn

    def __call__(self, args, **kwargs):
        wave = self.wave_gen(*args, **kwargs)
        if isinstance(wave, list):
            window = xp.asarray(get_window(self.window_fn, len(wave[0])))
            wave = [wave[i] * window for i in range(len(wave))]
        else:
            window = xp.asarray(get_window(self.window_fn, len(wave)))
            wave = wave * window

        return wave   


def get_tdi_generator(
        args,
        wave_gen,
):

    N_obs = int(args.tobs * YRSID_SI / args.dt)
    args.tobs = N_obs * args.dt / YRSID_SI

    orbit_file_esa = "esa-trailing-orbits.h5" 
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen = "2nd generation" if args.tdi2 else "1st generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 10000.0  # throw away on both ends when our orbital information is weird

    resp_gen = ResponseWrapper(
        wave_gen,
        args.tobs,
        args.dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=True,#"zero",  # removes the beginning of the signal that has bad information
        #n_overide=int(1e5),  # override the number of points (should be larger than the number of points in the signal)
        **tdi_kwargs_esa,
    )

    resp_gen = wave_gen_windowed(resp_gen, window_fn=('tukey', 0.005))

    return resp_gen

def get_snr2_single(pars, args, wf_gen, emri_kwargs, noise_psd):
    """
    
    """

    #breakpoint()
    injection = [pars[key] for key in PARS_NAMES]

    data_channels = wf_gen(injection, **emri_kwargs)
    fft_freq = xp.fft.rfftfreq(len(data_channels[0]),args.dt)    

    TDIA = xp.fft.rfft(data_channels[0]) * args.dt
    TDIE = xp.fft.rfft(data_channels[1]) * args.dt

    mask = fft_freq > args.freqs[0]

    snr = compute_snr2(fft_freq[mask], TDIA[mask], TDIE[mask], noise_psd)

    return snr

def get_snr2(pars, args, seed, traj, emri_kwargs, wf_gen, noise_psd):
    """
    Generate a waveform and compute the SNR
    """
    np.random.seed(seed)
    xp.random.seed(seed)

    pars_inj = pars.copy()

    p0 = get_p0(traj, pars, args.tobs)
    logger.info("new p0=%s", p0)
    pars_inj.update({'p0': p0})

    pars_here = pars_inj.copy()

    output = []
    
    for _ in range(args.ntrials):
        if args.randomize:
            randomize(pars_here, args)

        try:
            snr2 = get_snr2_single(pars_here, args, wf_gen, emri_kwargs, noise_psd) #already numpy
            snr = np.sqrt(snr2)
            point = [snr]
        except Exception as e:
            logger.error("Failed to generate waveform: %s", e)
            point = [None]

        output.append(point)
    
    output = np.concatenate(output)
    
    return output

def get_horizon_z(M, mu, snr, args, noise_psd):
    """
    Get the horizon redshift for a given (M, mu) point
    """
    if args.fixed_q:
        d_L = snr / args.snr_thr
        z = get_redshift(d_L)
        point_z = [M / (1+z), mu / (1+z), z] 
    else:
        raise NotImplementedError("Horizon search for fixed secondary mass not implemented yet.")
    return point_z

def get_from_outfile_z(_mu0, _M0, _file, _args):

    with open(_file, 'rb') as _fhand:
        data = pkl.load(_fhand)

    data = [item for item in data if item[0] == _M0 and item[1] == _mu0][0]

    if data[-1] is None:
        logger.warning("No horizon z search data found for (mu, M)=(%s, %s)", _mu0, _M0)
    else:
        logger.info("Found horizon z search data found for (mu, M)=(%s, %s)", _mu0, _M0)
        _args.z = data[-1]

    return [data[:-1]]


def get_from_outfile(_mu0, _M0, _file):

    with open(_file, 'rb') as _fhand:
        data = pkl.load(_fhand)

    data = [item for item in data if item[0] == _M0 and item[1] == _mu0]

    logger.info("Found %d entries in data file for for (mu, M)=(%s, %s)", len(data), _mu0, _M0)

    return data


if __name__ == "__main__":
    parser = lisa_parser()
    parser.add_argument('--report',  default="outputReport.md",
                        help="Path to markdown report")
    parser.add_argument('--outdir',  default="so3-horizon-unmerged",
                        help="Output subdirectory")
    parser.add_argument('--tobs', type=float, default=DEF_TOBS,
                        help=f"Observation time (yr) (def: {DEF_TOBS})")
    parser.add_argument('--dt', type=float, default=DEF_DT,
                        help=f"Time bin (def: {DEF_DT})")
    parser.add_argument('--redshift', dest='z', type=float, default=DEF_Z,
                        help=f"Default redshift (def: {DEF_Z})")
    parser.add_argument('--snr-thr', type=float, default=DEF_SNR_THR,
                        help=f"Threshold snr (def: {DEF_SNR_THR})")
    parser.add_argument('--maxit', type=int, default=75,
                        help="Maximum number of iterations.")
    parser.add_argument('--ntrials', type=int, default=100,
                        help="Number of generated sources")
    parser.add_argument('--start', type=int, default=0,
                        help="First (M, mu) grid point to process")
    parser.add_argument('--end', type=int, default=-1,
                        help="Last (M, mu) grid point to process")
    parser.add_argument('--no-random', dest='randomize', action='store_false',
                        help="Suppress generation of random parameters")
    parser.add_argument('--randomize-e', action='store_true', default=False,
                        help="Randomize the eccentricity")
    parser.add_argument('--no-horizon', dest='horizon', action='store_false',
                        help="Suppress the search for the horizon z")
    parser.add_argument('--no-outdata', dest='outdata', action='store_false',
                        help="DO not output the data file.")
    parser.add_argument('--fixed-pars', type=str, default=None,
                        help="Pickle file with fixed sky parameters per mu.")
    parser.add_argument('--fom-title', type=str, default="Horizon of EMRI",
                        help= "Title of the report")
    parser.add_argument('--fom-config', type=str, default="",
                        help= "Config file gathering FoM parameters")
    parser.add_argument('--nproc', type=int, default=-1,
                        help="Number of process")
    parser.add_argument('--gpu', action='store_true',
                        help="Use GPU (def: False)")
    parser.add_argument('--foreground', action='store_true', default=False,
                        help="Include the WD confusion foreground")
    parser.add_argument('--fixed_q', action='store_true', default=False,
                        help="Fixed mass ratio")
    

    #todo add the arguments here

    args = parser.parse_args()
    args = process_args(args)

    use_gpu = args.gpu

    args.duration = args.tobs * u.yr

    # Apply duty cycle to the SNR threshold (increasing it)
    args.snr_thr /= np.sqrt(args.duty_cycle)

    mass_grid = GRID_POINTS_Q[args.start: args.end] if args.fixed_q else GRID_POINTS_MU[args.start: args.end]

    lisa_arm_km = args.armlength.to("km").value

    logger = init_logger(args.log)
    outdir = os.path.join(os.path.dirname(args.report), args.outdir)
    logger.info("Running on the following (M, mu) grid points: %s", mass_grid)

    outfile = os.path.join(outdir, f'so3-horizon-data.{args.start}_{args.end}.pkl')
    outfile_z = os.path.join(outdir, f'so3-horizon-z.{args.start}_{args.end}.pkl')
    flagfile = os.path.join(outdir, f'done.{args.start}_{args.end}.flag')


    noise = build_lisa_noise(args, logger)
    if args.foreground:
        wd = args.tobs 
        logger.info("Adding the WD confusion foreground with Tobs=%s years", wd)
    else:
        wd = 0.0

    noise.set_wdconfusion(wd)

    noise_psd =  spline(xp.asarray(noise.freq), xp.asarray(noise.psd(noise.freq, option='A', tdi2=args.tdi2)))

    sum_kwargs = {
            "use_gpu": use_gpu, # GPU is available for this type of summation
            "pad_output": True
        }

    inspiral_kwargs={
            "err": 1e-10,
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e4),  # dense stepping trajectories
            "func":  TRAJECTORY
        }
    
    waveform_kwargs = {
            "T": args.tobs,
            "dt": args.dt,
        }
    
    traj = EMRIInspiral(func=TRAJECTORY)
    wave_gen = GenerateEMRIWaveform(
        *WAVEFORM_ARGS,
        inspiral_kwargs=inspiral_kwargs,
        sum_kwargs=sum_kwargs,
        return_list=False,
        use_gpu=use_gpu,
        frame='detector'
    )

    response_gen = get_tdi_generator(args, wave_gen)

    snr_vec = []
    horizon_vec = []

    ONLY_HORIZON = False
    if os.path.exists(outfile):
        logger.info("Data file exists. Only re-running search for horizon sources.")
        ONLY_HORIZON = True

    CONT_HORIZON = False
    if ONLY_HORIZON and os.path.exists(outfile_z):
        logger.info("Horizon z file also exists. Restarting iterations from there when possible.")
        CONT_HORIZON = True

    start_seed = args.start

    #todo copied and pasted from here, still need to understand what is happening

    for ind, (M, mu) in enumerate(mass_grid):

        # pylint: disable=invalid-name
        point_vec = None

        if CONT_HORIZON:
            point_vec = get_from_outfile_z(mu, M, outfile_z, args) ##todo define these functions

        if ONLY_HORIZON and point_vec is None:
            point_vec = get_from_outfile(mu, M, outfile)

        if not ONLY_HORIZON:
            logger.info("Started processing (M, mu)=(%s, %s)", M, mu)
            tracemalloc.start()
            itime = time()
            pars = get_def_pars()
            pars.update({'M': M, 'mu': mu})

            point_vec = get_snr2(
                pars, args,
                seed=start_seed + ind,
                traj=traj,
                emri_kwargs=waveform_kwargs,
                wf_gen=response_gen,
                noise_psd=noise_psd
            )
            snr_vec += [point_vec]
            logger.info("Finished processing (M, mu)=(%s, %s); etime=%s; ram=%s",
                        M, mu, time() - itime, tracemalloc.get_traced_memory())
            tracemalloc.stop()

        if not args.horizon:
            continue

        
        logger.info("Getting the horizon z for (M, mu)=(%s, %s)", M, mu)
        try:
            average_snr = np.mean(point_vec[point_vec != None])
        except Exception as e:
            logger.error("Failed to get the average SNR: %s", e)
            
            continue
        try:
            point_z = get_horizon_z(M, mu, average_snr, args, noise_psd)
            horizon_vec.append(point_z)
            logger.info("Finished finding horizon z for (M, mu)=(%s, %s)", M, mu)

        # pylint: disable=broad-except
        except Exception:
            logger.error(traceback.format_exc())
            logger.warning("Failed to get the horizon z for (M, mu)=(%s, %s)", M, mu)
            horizon_vec.append([M, mu, None])

    # outputs
    if args.outdata and not ONLY_HORIZON:
        logger.info("Saving data. Output file: %s", outfile)
        os.makedirs(outdir, exist_ok=True)
        with open(outfile, 'wb') as fhand:
            pkl.dump(snr_vec, fhand)

    if args.horizon:
        logger.info("Saving horizon_z. Output file: %s", outfile_z)
        with open(outfile_z, 'wb') as fhand:
            pkl.dump(horizon_vec, fhand)

        # Dealing with a flag file instead of the input, in Snakefile
        # allows for restaring horizon search to the last saved iteration
        logger.info("Creating the flag file %s", flagfile)
        with open(flagfile, 'w') as _:
            pass

    
close_logger(logger)