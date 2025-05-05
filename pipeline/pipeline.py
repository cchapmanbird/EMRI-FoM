# python pipeline.py --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 1.0 --z 0.1 --repo test --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 3 --repo test
import os
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform
from few.utils.geodesic import get_separatrix
from few.trajectory.ode import KerrEccEqFlux
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from stableemrifisher.fisher import StableEMRIFisher
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from fastlisaresponse import ResponseWrapper
from common import standard_cosmology
import time
import matplotlib.pyplot as plt
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot

#psd stuff
from psd_utils import load_psd, get_psd_kwargs

# Initialize logger
logger = logging.getLogger()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--M", help="Mass of the central black hole", type=float)
    parser.add_argument("--mu", help="Mass of the compact object", type=float)
    parser.add_argument("--a", help="Spin of the central black hole", type=float)
    parser.add_argument("--e_f", help="Final eccentricity", type=float)
    parser.add_argument("--T", help="Time to coalescence", type=float)
    parser.add_argument("--z", help="Redshift", type=float)
    parser.add_argument("--repo", help="Name of the folder where results are stored", type=str)
    parser.add_argument("--psd_file", help="Path to a file containing PSD frequency-value pairs", default="TDI2_AE_psd.npy")
    parser.add_argument("--dt", help="Sampling cadence in seconds", type=float, default=10.0)
    parser.add_argument("--use_gpu", help="Whether to use GPU for FIM computation", action="store_true")
    parser.add_argument("--N_montecarlo", help="How many random sky localizations to generate", type=int, default=10)
    parser.add_argument("--device", help="GPU device", type=int, default=0)
    parser.add_argument('--foreground', action='store_true', default=False, help="Include the WD confusion foreground")
    parser.add_argument('--esaorbits', action='store_true', default=False, help="Use ESA trailing orbits. Default is equal arm length orbits.")
    parser.add_argument('--tdi2', action='store_true', default=False, help="Use 2nd generation TDI channels")
    parser.add_argument('--channels', type=str, default="AE", help="TDI channels to use")
    parser.add_argument('--model', type=str, default="scirdv1", help="Noise model to use")
    
    return parser.parse_args()

def initialize_gpu(args):
    if args.use_gpu:
        import cupy as xp
        print("Using GPU", args.device)
        xp.cuda.Device(args.device).use()
        xp.random.seed(2601)
    else:
        xp = np
    np.random.seed(2601)
    return xp

# def load_psd(psd_file):
#     psdf, psdv = np.load(psd_file).T
#     min_psd = np.min(psdv)
#     max_psd = np.max(psdv)
#     print("PSD range", min_psd, max_psd)
#     psd_interp = CubicSplineInterpolant(psdf, psdv)
#     def psd_clipped(f, **kwargs):
#         f = np.clip(f, 0.00001, 1.0)
#         return np.clip(psd_interp(f), min_psd, max_psd)

#     return psd_clipped


def initialize_waveform_generator(T, args, inspiral_kwargs_forward):
    backend = 'gpu' if args.use_gpu else 'cpu'
    base_wave = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", inspiral_kwargs=inspiral_kwargs_forward, force_backend=backend, sum_kwargs=dict(pad_output=True))
    tdi_kwargs_esa, orbits = initialize_tdi_generator(args)
    model = ResponseWrapper(
            base_wave, T, args.dt, 8, 7, t0=100000., flip_hx=True, use_gpu=args.use_gpu, orbit=orbits,
            remove_sky_coords=False, is_ecliptic_latitude=False, remove_garbage=True, **tdi_kwargs_esa
        )
    return model

def initialize_tdi_generator(args):
    # orbits = "esa-trailing-orbits.h5" if args.esaorbits else "equalarmlength-orbits.h5"
    orbits = ESAOrbits(use_gpu=args.use_gpu) if args.esaorbits else EqualArmlengthOrbits(use_gpu=args.use_gpu)
    # orbit_file = os.path.join(os.path.dirname(__file__), '..', 'lisa-on-gpu', 'orbit_files', orbits)
    # orbit_kwargs = dict(orbit_file=orbit_file)
    # tdi_kwargs = dict(orbit_kwargs=orbit_kwargs, order=25, tdi="2nd generation", tdi_chan="AET")
    tdi_kwargs = dict(order=25, tdi="2nd generation", tdi_chan="AET")
    return tdi_kwargs, orbits

def generate_random_phases():
    return np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)

def generate_random_sky_localization():
    qS = np.pi/2 - np.arcsin(np.random.uniform(-1, 1))
    phiS = np.random.uniform(0, 2 * np.pi)
    qK = np.pi/2 - np.arcsin(np.random.uniform(-1, 1))
    phiK = np.random.uniform(0, 2 * np.pi)
    return qS, phiS, qK, phiK

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

inspiral_kwargs_back = {"err": 1e-10,"integrate_backwards": True}
inspiral_kwargs_forward = {"err": 1e-10,"integrate_backwards": False}

param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])
popinds = []
popinds.append(5)
popinds.append(12)
param_names = np.delete(param_names, popinds).tolist()

if __name__ == "__main__":

    args = parse_arguments()
    #args = process_args(args)
    xp = initialize_gpu(args)
    
    # create repository
    os.makedirs(args.repo, exist_ok=True)

    # load psd
    #psd_wrap = load_psd(args.psd_file)
    custom_psd_kwargs = {
            'tdi2': args.tdi2,
            'channels': args.channels,
        }
    
    if args.foreground:
        custom_psd_kwargs["stochastic_params"] = (args.T * YRSID_SI,)
        custom_psd_kwargs["include_foreground"] = True
    psd_kwargs = get_psd_kwargs(custom_psd_kwargs)

    psd_wrap = load_psd(logger=logger, filename=args.psd_file, xp=xp, **psd_kwargs)
    
    # get the detector frame parameters
    M = args.M * (1 + args.z)
    mu = args.mu * (1 + args.z)
    a = args.a
    e_f = args.e_f
    x0_f = 1.0
    p_f = get_separatrix(args.a, args.e_f, x0_f) + 0.1
    dist = standard_cosmology(H0=67.).dl_zH0(args.z) / 1000.
    T = args.T
    # `source_frame_data` is a dictionary that contains various parameters related to the source frame
    detector_frame_data = {
        "M central black hole mass": M,
        "mu secondary black hole mass": mu,
        "a dimensionless central object spin": a,
        "p_f final semi-latus rectum": p_f,
        "e_f final eccentricity": e_f,
        "z redshift": args.z,
        "dist luminosity distance in Gpc": dist,
        "T inspiral duration in years": T,
    }
    source_frame_data = {
        "M central black hole mass": args.M,
        "mu secondary black hole mass": args.mu,
        "a dimensionless central object spin": args.a,
        "p_f final semi-latus rectum": p_f,
        "e_f final eccentricity": args.e_f,
        "z redshift": args.z,
        "dist luminosity distance in Gpc": dist,
        "T inspiral duration in years": T,
    }
    # save in the repository the source and detector frame parameters
    for el,name in zip([detector_frame_data, source_frame_data], ["detector_frame_data", "source_frame_data"]):
        df = pd.DataFrame(el, index=[0])
        # save df using pandas
        df.to_markdown(os.path.join(args.repo, f"{name}.md"), floatfmt=".10e")
        # save df using npz
        np.savez(os.path.join(args.repo, f"{name}.npz"), **el)

    # initialize the trajectory
    traj = EMRIInspiral(func=KerrEccEqFlux)
    print("Generating backward trajectory")
    t_back, p_back, e_back, x_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(M, mu, a, p_f, e_f, x0_f, dt=1e-4, T=T, integrate_backwards=True)
    print("Done with the trajectory")
    # initialiaze the waveform generator
    model = initialize_waveform_generator(T, args, inspiral_kwargs_forward)
    # save in the repository the source and detector frame parameters
    # define the initial parameters
    p0, e0, x0 = p_back[-1], e_back[-1], x_back[-1]
    print("p0, e0, x0", p0, e0, x0)
    Phi_phi0, Phi_r0, Phi_theta0 = generate_random_phases()
    qS, phiS, qK, phiK = generate_random_sky_localization()
    parameters = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])
    model(*parameters)

    tic = time.time()
    waveform_out = model(*parameters)
    toc = time.time()
    timing = toc - tic
    print("Time taken for one waveform generation: ", timing)
    print("\n")
    # save the waveform generation time
    with open(os.path.join(args.repo, "waveform_generation_time.txt"), "w") as f:
        f.write(str(timing))
    # check if there are nans in the waveform_out[0]
    if xp.isnan(xp.asarray(waveform_out)).any():
        print("There are nans in the waveform")
    # plot the waveform in the frequency domain
    # window the signal using scipy.signal.windows.tukey
    from scipy.signal.windows import tukey
    window = xp.asarray(tukey(len(waveform_out[0]), alpha=0.05))
    fft_waveform = xp.fft.rfft(waveform_out[0]*window).get() *args.dt
    freqs = np.fft.rfftfreq(len(waveform_out[0]), d=args.dt)
    mask = (freqs>1e-4)
    plt.figure()
    plt.loglog(freqs[mask], np.abs(fft_waveform)[mask]**2)
    plt.loglog(freqs[mask], np.atleast_2d(psd_wrap(freqs[mask]).get())[0], label="PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude $|\tilde h(f)|$")
    plt.legend()
    plt.savefig(os.path.join(args.repo, "waveform.png"))

    if args.e_f < 1e-3:
        log_e = True
    else:
        log_e = False
    
    if log_e:
        EMRI_waveform_gen = transf_log_e_wave(model)
    else:
        EMRI_waveform_gen = model
    
    deltas = None
    # start loop over multiple realizations
    for j in range(args.N_montecarlo):
        print("--------------------------------------")
        print(f"Generating source {j} realization")
        name_realization = f"realization_{j}"
        # generate random parameters
        Phi_phi0, Phi_r0, Phi_theta0 = generate_random_phases()
        qS, phiS, qK, phiK = generate_random_sky_localization()
        # define the initial parameters
        parameters = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])
        # create folder for the realization
        current_folder = os.path.join(args.repo, name_realization)
        os.makedirs(current_folder, exist_ok=True)
        # save the parameters to txt file
        np.savetxt(os.path.join(current_folder, "all_parameters.txt"), parameters.T, header=" ".join(param_names))

        fish = StableEMRIFisher(*parameters, 
                                dt=args.dt, T=T, EMRI_waveform_gen=EMRI_waveform_gen, noise_model=psd_wrap, noise_kwargs=dict(TDI="TDI2"), param_names=param_names, stats_for_nerds=False, use_gpu=args.use_gpu, 
                                der_order=4., Ndelta=20, filename=current_folder,
                                deltas = deltas,
                                log_e = log_e, # useful for sources close to zero eccentricity
                                CovEllipse=False, # will return the covariance and plot it
                                stability_plot=False, # activate if unsure about the stability of the deltas
                                window=window # addition of the window to avoid leakage
                                )
        #execution
        SNR = fish.SNRcalc_SEF()
        fim = fish()
        cov = np.linalg.inv(fim)
        fish.save_deltas()
        # check the inversion
        print("if correct matrix inversion, then",np.diag(fim @ cov).sum() - fim.shape[0], "should be approximately zero")
        # check dimensions
        print("Fisher matrix shape", fim.shape[0]==len(param_names))
        if log_e:
            jac = np.diag([1, 1, 1, 1, 1/parameters[4], 1, 1, 1, 1, 1, 1, 1]) #if working in log_e space apply jacobian to the fisher matrix
            fim = jac.T @ fim @ jac

        if deltas is None:
            deltas = fish.deltas
        
        # create ellipse plot only the first montecarlo realization
        cov = np.linalg.inv(fim)
        if j == 0:
            CovEllipsePlot(fish.param_names, fish.wave_params, cov, filename=current_folder + f"/covariance_ellipse_plot.png")
        
        # get errors
        errors = np.sqrt(np.diag(cov))
        # save the errors with pandas to markdown
        fisher_params = np.delete(parameters, popinds)
        errors_df = {"Parameter": param_names, "parameter value": fisher_params, "1 sigma Error": errors, "Relative Error": errors/fisher_params, "SNR": SNR}
        errors_df = pd.DataFrame(errors_df)
        errors_df.to_markdown(os.path.join(current_folder, "summary.md"), floatfmt=".10e")
        # save the covariance matrix and the SNR to npz file
        np.savez(os.path.join(current_folder, "results.npz"), cov=cov, snr=SNR, fisher_params=fisher_params, errors=errors, relative_errors=errors/fisher_params, names=param_names)
        print("Saved results to", current_folder)
        print("*************************************")



