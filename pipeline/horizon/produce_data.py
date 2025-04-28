# Description: Script to generate horizon data for a given set of parameters
# python produce_horizon_data.py -Tobs 2.0 -dt 5.0 -t kerr kerr pn5 -wf kerr aak aak -outname test -grids e0 M -qs 1e-5 --tdi2 --foreground --esaorbits

import argparse
import os
import GPUtil
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
print("PID:",os.getpid())
import time
parser = argparse.ArgumentParser(description="horizon redshift")
parser.add_argument('--tdi2', action='store_true', default=False, help="Use 2nd generation TDI channels")
parser.add_argument('--channels', type=str, default="AET", help="TDI channels to use")
parser.add_argument('--foreground', action='store_true', default=False, help="Include the WD confusion foreground")
parser.add_argument('--esaorbits', action='store_true', default=False, help="Use ESA trailing orbits. Default is equal arm length orbits.")
parser.add_argument('--model', type=str, default="scirdv1", help="Noise model")
parser.add_argument("-dev", "--dev", help="GPU device", required=False, type=int, default=None)
parser.add_argument("-Tobs", "--Tobs", help="Observation Time in years", required=False, default=1.0, type=float)
parser.add_argument("-Ms", "--Ms", help="masses", required=False, nargs='*', default=1e6, type=float)
parser.add_argument("-qs", "--qs", help="mass ratio", required=False, nargs='*', default=5e-5, type=float)
parser.add_argument("-e0s", "--e0s", help="initial eccentricity", required=False, nargs='*', default=0.0, type=float)
parser.add_argument("-spins", "--spins", help="dimensionless spin", required=False, nargs='*', default=0.99, type=float)
parser.add_argument("-dt", "--dt", help="sampling interval delta t", required=False, type=float, default=5.0)
parser.add_argument("-SNR", "--SNR", help="SNR", required=False, type=float, default=20.0)
parser.add_argument("-outname", "--outname", help="output name", required=False, type=str, default="")
parser.add_argument("-t", "--traj", help="trajectory", required=False, nargs='*', type=str, default="kerr")
parser.add_argument("-wf", "--wf", help="waveform", required=False, nargs='*', type=str, default="kerr")
parser.add_argument("-grids", "--grids", help="parameters to iterate over", required=False, nargs=2, default=['e0', 'M'], type=str)
parser.add_argument("-avg_n", "--avg_n", help="number of samples to average over", required=False, type=int, default=100)


args = vars(parser.parse_args())

dev = args['dev']

import sys, os

import matplotlib.pyplot as plt

import matplotlib.lines as mlines
import numpy as np
from eryn.prior import ProbDistContainer, uniform_dist

#sys.path.append('/data/asantini/emris/DirtyEMRI/DataAnalysis/LISAanalysistools/')

from lisatools.diagnostic import *
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits

from few.trajectory.ode.flux import SchwarzEccFlux, KerrEccEqFlux
from few.trajectory.ode.pn5 import PN5
from few.waveform.waveform import GenerateEMRIWaveform, AAKWaveformBase, FastKerrEccentricEquatorialFlux, FastSchwarzschildEccentricFlux
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.utils.constants import *
from few.utils.utility import get_p_at_t
from few.utils.utility import get_fundamental_frequencies #! change to few.utils.geodesic if using the last version of few
from few.utils.globals import get_first_backend

from fastlisaresponse import ResponseWrapper

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

from psd_utils import load_psd, get_psd_kwargs, compute_snr2
import logging

import h5py

from scipy.signal import get_window
from tqdm import tqdm

SEED = 26011996

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
    # set GPU device
    if dev is None:
        free_gpus = get_free_gpus(n_gpus=1)
        if not free_gpus:
            gpu_available = False
        else:
            dev = free_gpus[0]
            gpu_available = True
    else:
        os.system("CUDA_VISIBLE_DEVICES="+str(dev))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
        print("Using GPU", dev)
        gpu_available = True
    
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"

import warnings
warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = True

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

np.random.seed(SEED)
xp.random.seed(SEED)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

import jax.numpy as jnp
from jax import jit, vmap



cosmo = Planck18 #FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def get_redshift(distance):
    return (z_at_value(cosmo.luminosity_distance, distance * u.Gpc )).value

def get_distance(redshift):
    return cosmo.luminosity_distance(redshift).to(u.Gpc).value

class wave_gen_windowed:
    """
    Generate a waveform and apply a window to it
    """
    def __init__(self, wave_gen, window_fn=('tukey', 0.005)):
        self.wave_gen = wave_gen
        self.window_fn = window_fn

    def __call__(self, *args, **kwargs):
        wave = self.wave_gen(*args, **kwargs)
        if isinstance(wave, list):
            window = xp.asarray(get_window(self.window_fn, len(wave[0])))
            wave = [wave[i] * window for i in range(len(wave))]
        else:
            window = xp.asarray(get_window(self.window_fn, len(wave)))
            wave = wave * window

        return wave   

@jit
def compute_Omega_mn(omegaPhi, omegaR, m, n):
    """
    Compute the Omega_mn frequencies.
    """
    return m*omegaPhi + n*omegaR

@jit
def compute_Omega_mn_vectorized(omegaPhi, omegaR, m_array, n_array):
    """
    Compute all combinations of Omega_mn frequencies.
    Returns a matrix of shape (len(m_array), len(n_array))
    """
    # Create a function that maps over m for fixed n
    f1 = vmap(lambda m, n: compute_Omega_mn(omegaPhi, omegaR, m, n), in_axes=(0, None))
    
    # Then map that function over all n values
    return vmap(lambda n: f1(m_array, n))(n_array)

def find_maximum_frequency(
        waveform_generator,
        M,
        mu,
        spin,
        p0,
        e0,
        x0,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        T,
        dt,
        buffer_time_seconds=3600,
        inspiral_kwargs={},
):
    if 'T' not in inspiral_kwargs:
        inspiral_kwargs['T'] = T
    if 'dt' not in inspiral_kwargs:
        inspiral_kwargs['dt'] = dt

    (t, p, e, xI, Phi_phi, Phi_theta, Phi_r) = waveform_generator.inspiral_generator(
            M,
            mu,
            spin,
            p0,
            e0,
            x0,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            **inspiral_kwargs,
        )
    
    t_index = np.where(t >= t[-1] - buffer_time_seconds)[0][0] 
    omegaPhi, omegaTheta, omegaR = get_fundamental_frequencies(spin, p[t_index], e[t_index], xI[t_index])

    dimension_factor = 2.0 * np.pi * M * MTSUN_SI
    omegaPhi = omegaPhi / dimension_factor
    omegaTheta = omegaTheta / dimension_factor
    omegaR = omegaR / dimension_factor

    ns = jnp.asarray(waveform_generator.ns)
    ms = jnp.asarray(waveform_generator.ms)

    Omega_mn = compute_Omega_mn_vectorized(omegaPhi, omegaR, ms, ns)

    maximum_frequency = jnp.max(Omega_mn)

    return maximum_frequency

def generate_data(
        args,
        fixed_params,
        grids,
        outname,
        traj,
        wave_gen,
        psd_fn,
        avg_n=1,
        snr_thr=20.0,
        emri_kwargs={},
        plot_waveform=False,
        ):
    """
    Generate data for horizon plot
    """

    FMIN, FMAX = 2e-9, 1.0
    DT_LOWMASS = 2.0

    Tobs, dt = args['Tobs'], args['dt']
    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI

    tdi_gen = "2nd generation" if args['tdi2'] else "1st generation"# or "2nd generation"

    order = 25  # interpolation order (should not change the result too much)
    orbits = ESAOrbits(use_gpu=use_gpu) if args['esaorbits'] else EqualArmlengthOrbits(use_gpu=use_gpu)
    
    orbit_file = orbits.filename
    orbit_file_kwargs = dict(orbit_file=orbit_file)


    tdi_kwargs_esa = dict(
        #orbits=orbits,
        orbit_kwargs=orbit_file_kwargs,
        order=order,
        tdi=tdi_gen,
        tdi_chan=args['channels'],
    )  # could do "AET

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 10000.0  # throw away on both ends when our orbital information is weird

    window_args = ('tukey', 0.005)  # window function to apply to the waveform

    resp_gen_custom = ResponseWrapper(
        wave_gen,
        Tobs,
        dt,
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

    # resp_gen_lowmass = ResponseWrapper(
    #     wave_gen,
    #     Tobs,
    #     DT_LOWMASS,
    #     index_lambda,
    #     index_beta,
    #     t0=t0,
    #     flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
    #     use_gpu=use_gpu,
    #     is_ecliptic_latitude=False,  # False if using polar angle (theta)
    #     remove_garbage=True,#"zero",  # removes the beginning of the signal that has bad information
    #     #n_overide=int(1e5),  # override the number of points (should be larger than the number of points in the signal)
    #     **tdi_kwargs_esa,
    # )

    resp_gen_lowmass = resp_gen_custom

    resp_gen_custom = wave_gen_windowed(resp_gen_custom, window_fn=window_args)
    resp_gen_lowmass = wave_gen_windowed(resp_gen_lowmass, window_fn=window_args)


    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(-0.99999, 0.99999),  # qS
                1: uniform_dist(0.0, 2 * np.pi),  # phiS
                2: uniform_dist(-0.99999, 0.99999),  # qK
                3: uniform_dist(0.0, 2 * np.pi),  # phiK
                4: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                5: uniform_dist(0.0, 2 * np.pi),  # Phi_theta0
                6: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    def get_p0(M, mu, a, e0, x0, Tobs):
        # fix p0 given T
        left_bound = None#get_separatrix(a,e0,x0)+0.1
        right_bound = 200.0
        try:
            #try:
            p0 = get_p_at_t(traj, Tobs * 0.999, [M, mu, a, e0, x0], bounds=[left_bound, right_bound])
            # except:
            #     left_bound = max(left_bound, 3.41)
            #     p0 = get_p_at_t(traj, Tobs * 0.999, [M, mu, a, e0, x0], bounds=[left_bound, right_bound])
        except Exception as e:
            logger.info(e)
            p0 = None   
        #logger.info("new p0 fixed by Tobs, p0=", p0, traj(M, mu, a, p0, e0, x0, T=10.0)[0][-1]/YRSID_SI)
        return p0
    
    def zero_pad(data):
        """
        Inputs: data stream of length N
        Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
        """
        N = len(data)
        pow_2 = xp.ceil(np.log2(N))
        return xp.pad(data,(0,int((2**pow_2)-N)),'constant')
    
    def get_snr(inp, psd_fn, maximum_frequency=None, emri_kwargs={}):
        data_channels = resp_gen(*inp, kwargs=emri_kwargs)
    
        data_channels_padded= [zero_pad(item) for item in data_channels]
        data_channels_fft = xp.array([xp.fft.rfft(item) * emri_kwargs['dt'] for item in data_channels_padded])
        freqs = xp.fft.rfftfreq(len(data_channels_padded[0]),emri_kwargs['dt'])
        fmax = maximum_frequency if maximum_frequency is not None else FMAX
        mask = (freqs > FMIN) & (freqs < fmax)
        data_channels_fft = data_channels_fft[:, mask]
        freqs = freqs[mask]

        snr2 = compute_snr2(freqs, data_channels_fft, psd_fn, xp=xp)

        return xp.sqrt(snr2)
  
    
    def get_snr_avg(M, mu, spin, e0, x0, Tobs, psd_fn, avg_n=1, emri_kwargs={}):
        p0 = get_p0(M, mu, spin, e0, x0, Tobs)
        if p0 is None:
            return np.nan, np.nan #return nan if p0 is not found
        prior_draw = priors['emri'].rvs(avg_n) #random draw from prior for the extrinsic parameters

        prior_draw[:, 0] = np.arccos(prior_draw[:, 0])  # qS
        prior_draw[:, 2] = np.arccos(prior_draw[:, 2]) # qK

        injection = np.empty((avg_n, 7))
        injection[:, 0] = M
        injection[:, 1] = mu
        injection[:, 2] = spin
        injection[:, 3] = p0
        injection[:, 4] = e0
        injection[:, 5] = x0  
        injection[:, 6] = 1.0 #distance
        
        injection = np.concatenate((injection, prior_draw), axis=1)

        data_channels = resp_gen(*injection[0], kwargs=emri_kwargs)
        maximum_frequency = find_maximum_frequency(
            wave_gen.waveform_generator,
            M,
            mu,
            spin,
            p0,
            e0,
            x0,
            prior_draw[0, 4],
            prior_draw[0, 5],
            prior_draw[0, 6],
            Tobs,
            dt,
            buffer_time_seconds=3600,
            inspiral_kwargs=emri_kwargs,
        )
        logger.info(f"Maximum frequency: {maximum_frequency}")
        
        if plot_waveform:
            nchannels = len(data_channels)

            ffth = [xp.fft.rfft(data_channels[i])*dt for i in range(nchannels)]
            fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)
            fft_freq[0] = fft_freq[1] # remove the zero frequency

            mask = (fft_freq > FMIN) & (fft_freq < FMAX)
            ffth = [ffth[i][mask] for i in range(nchannels)]
            fft_freq = fft_freq[mask]

            PSD_arr = xp.atleast_2d(psd_fn(fft_freq))
            
            fig, axs = plt.subplots(1, nchannels, figsize=(15, 5), sharex=True)
            fig.suptitle(f"PSD and FFT for M={M:.1e}, mu={mu:.1e}, a={spin:.1e}, e0={e0:.1e}, x0={x0:.1e}")
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            fig.tight_layout()
            for i in range(nchannels):
                try:
                    axs[i].plot(fft_freq.get(), (xp.abs(ffth[i])**2).get())
                    axs[i].loglog(fft_freq.get(), PSD_arr[i].get())
                except:
                    axs[i].plot(fft_freq, (xp.abs(ffth[i])**2))
                    axs[i].loglog(fft_freq, PSD_arr[i])
                axs[i].axvline(maximum_frequency, color='black', linestyle='--', label='Maximum frequency')
                axs[i].set_title(f"Channel {i}")
                axs[i].legend()
                axs[i].set_xlabel("Frequency [Hz]")
            axs[0].set_ylabel("Power [Hz$^{-1}$]")

            savepath = savename[:-3] + '_tmp_plots/' + f"M_{M:.1e}_mu_{mu:.1e}_a_{spin:.1e}_e0_{e0:.1e}_x0_{x0:.1e}.pdf"

            plt.savefig(savepath)
            plt.close(fig)
        
        snr_all = xp.array([get_snr(inp, psd_fn, maximum_frequency, emri_kwargs) for inp in injection])
        snr_here = xp.mean(snr_all)
        try:
            snr_here = snr_here.get()
            snr_all = snr_all.get()
        except:
            pass
        try:
            assert np.isfinite(snr_here)
        except AssertionError:
            breakpoint()
        logger.info(f"Average snr: {snr_here}")
        return snr_here, snr_all

    with h5py.File(outname, 'w') as f:
        f.attrs['Tobs'] = Tobs
        f.attrs['dt'] = dt
        for key, val in fixed_params.items():
            f.attrs[key] = val
        
        for key, val in grids.items():
            f.attrs[key] = val

        params_all = fixed_params.copy()

        keys_grid = list(grids.keys())
        key_lines, vals_lines = keys_grid[0], grids[keys_grid[0]]
        key_x, vals_x = keys_grid[1], grids[keys_grid[1]]

        mumin, mumax = 1e-2, 1e7
        
        for line_element in tqdm(vals_lines):
            msg = f"Calculating for {key_lines} = {line_element}"
            logger.info(msg)
            z = np.zeros((len(vals_x),))
            std = np.zeros((len(vals_x),))
            for i, x_element in enumerate(vals_x):
                params_all[key_lines] = line_element
                params_all[key_x] = x_element

                M = params_all['M']
                q = params_all['q']
                spin = params_all['spin']
                e0 = params_all['e0']
                #x0 = params_all['x0']
                x0 = np.sign(spin) * 1.0 if spin != 0.0 else 1.0
                spin = np.abs(spin)
                logger.info(f"M, q, spin, e0, x0, {M}, {q}, {spin}, {e0}, {x0}")
                mu = q * M
                if mu < mumin or mu > mumax:
                    z[i] = np.nan
                    continue
                            
                # if q >= 1e-3 and M > 7e7:  
                #     z[i] = np.nan
                #     continue

                # elif q >= 1e-2 and M > 3e7:
                #     z[i] = np.nan
                #     continue
                
                # deal with timestep
                emri_kwargs_here = emri_kwargs.copy()
                if M < 1e6:
                    emri_kwargs_here['dt'] = DT_LOWMASS
                    resp_gen = resp_gen_lowmass
                else:
                    resp_gen = resp_gen_custom

                snr_here, snr_all = get_snr_avg(M, mu, spin, e0, x0, Tobs, psd_fn=psd_fn, avg_n=avg_n, emri_kwargs=emri_kwargs_here)
            
                d_L = snr_here / snr_thr
                d_all = snr_all / snr_thr # use it to compute an uncertainty

                z[i] = get_redshift(d_L)
                z_all = get_redshift(d_all)
                std[i] = np.std(z_all)

                logger.debug(f"is F(mean snr) close to mean(F(snr))? {np.isclose(z[i], np.mean(z_all))}")

                logger.info(f"Horizon redshift: {z[i]} +/- {std[i]}")
                if not np.isclose(z[i], np.mean(z_all)):
                    logger.debug(f"Mean redshift: {np.mean(z_all)}")

            f.create_dataset(key_lines + f'_{line_element}', data=z)
            f.create_dataset(key_lines + f'_{line_element}_sigma', data=std)

if __name__ == '__main__':

    logger = logging.getLogger(name='horizon')
    level = logging.INFO
    logger.setLevel(level)
    if (len(logger.handlers) < 2):
        formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
        
        shandler = logging.StreamHandler(sys.stdout)
        shandler.setLevel(level)
        shandler.setFormatter(formatter)
        logger.addHandler(shandler)

    start_time = time.time()
    PLOT_WAVEFORM = False

    Tobs = args['Tobs']
    Ms = args['Ms']
    qs = args['qs']
    spins = args['spins']
    e0s = args['e0s']
    dt = args['dt']
    snr_thr = args['SNR']
    base_outname = args['outname']
    traj_module = args['traj']
    wf_module = args['wf']
    grid_keys = args['grids']

    inspiral_func_all = dict(zip(['kerr', 'schwarzschild', 'pn5'], [KerrEccEqFlux, SchwarzEccFlux, PN5]))
    args_all = dict(zip(['kerr', 'schwarzschild', 'aak'], [[FastKerrEccentricEquatorialFlux,], [FastSchwarzschildEccentricFlux,], [AAKWaveformBase, EMRIInspiral, AAKSummation]]))

    eps = 1e-5 # mode content percentage

    logger.info("generating different " + grid_keys[0] + " lines for " + grid_keys[1] + " grid")

    params_all = ['M', 'q', 'spin', 'e0']
    fixed_params_keys = [key for key in params_all if key not in grid_keys]
    fixed_params_all = {}
    for key, el in zip(params_all, [Ms, qs, spins, e0s]):
        if isinstance(el, float):
            el = [el]
        fixed_params_all[key] = el

    #e0_grid = np.arange(0.15, 0.76, 0.15)
    e0_grid = [0.01, 0.3, 0.6, 0.75]
    spin_grid = [0.0, 0.25, 0.5, 0.75, 0.99, 0.999]
    q_grid = [1e-6, 1e-5, 1e-4]

    Mmin, Mmax = 5e4, 5e8
    nmasses = 20

    M_grid =10**np.linspace(np.log10(Mmin), np.log10(Mmax), num=nmasses)

    grids_all = {
        'M': M_grid,
        'q': q_grid,
        'spin': spin_grid,
        'e0': e0_grid,
    }

    grids = {key: grids_all[key] for key in grid_keys}
    assert len(grids) == 2

    fixed_params_all.pop(grid_keys[0])
    fixed_params_all.pop(grid_keys[1])

    ## psd setup
    custom_psd_kwargs = {
        'tdi2': args['tdi2'],
        'channels': args['channels'],
    }

    if args['foreground']:
        custom_psd_kwargs['stochastic_params'] = (Tobs * YRSID_SI,)
        custom_psd_kwargs['include_foreground'] = True  

    psd_kwargs = get_psd_kwargs(custom_psd_kwargs)

    noise_psd = load_psd(logger=logger, filename=None, xp=xp, **psd_kwargs)


    for traj_here, amp_here in zip(traj_module, wf_module):
        start_section_time = time.time()
        assert traj_here in ['kerr', 'schwarzschild', 'pn5'], traj_here
        assert amp_here in ['kerr', 'schwarzschild', 'aak'], amp_here
        
        if traj_here == 'schwarzschild':
            fixed_params['spin'] = [0.0]

        outname = base_outname + '_traj_' + traj_here + '_wf_' + amp_here + '_'

        inspiral_func = inspiral_func_all[traj_here]
        traj = EMRIInspiral(func=inspiral_func)
        args_here = args_all[amp_here]


        best_backend = get_first_backend(FastKerrEccentricEquatorialFlux.supported_backends())
        backend = best_backend if use_gpu else 'cpu'

        sum_kwargs = {
            "force_backend": backend, # GPU is available for this type of summation
            "pad_output": True
        }

        inspiral_kwargs={
                "err": 1e-10,
                "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
                "max_init_len": int(1e4),  # dense stepping trajectories
                "func": inspiral_func
            }
        
        wave_gen = GenerateEMRIWaveform(
            *args_here,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            return_list=False,
            force_backend=backend,
            frame="detector"
        )
        waveform_kwargs = {
            "T": Tobs,
            "dt": dt,
        }

        if amp_here != 'aak':
            waveform_kwargs['eps'] = eps

        #wave_gen = wave_gen_windowed(wave_gen, window_fn=('tukey', 0.005))
        savename = './horizon/data/' + outname + 'T_%.1f' % Tobs
        fixed_params = {}
        
        key1, key2 = fixed_params_keys

        for val1 in fixed_params_all[key1]:
            fixed_params[key1] = val1
            for val2 in fixed_params_all[key2]:
                fixed_params[key2] = val2

                savename = savename + '_%s_%.1e' % (key1, val1) + '_%s_%.1e' % (key2, val2)
                os.makedirs(savename + '_tmp_plots', exist_ok=True)
                savename = savename + '.h5'

                if not os.path.exists(savename):
                    logger.info(f"Generating data for {traj_here} trajectory and {amp_here} waveform") 
                    try:
                        generate_data(
                            args,
                            fixed_params,
                            grids,
                            savename,
                            traj,
                            wave_gen,
                            noise_psd,
                            avg_n=args['avg_n'],
                            snr_thr=snr_thr,
                            emri_kwargs=waveform_kwargs,
                            plot_waveform=PLOT_WAVEFORM
                        )

                        fixed_params = {} #reset dictionary
                    except ValueError as e:
                        logger.info(e)
                        logger.info("Error in generating data")
                        logger.info(fixed_params)
                        if os.path.exists(savename):
                            logger.info("Removing file")
                            os.remove(savename)
                else:
                    logger.info("Data already exists, skipping")
            
        logger.info("done")