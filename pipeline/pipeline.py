#!/data/lsperi/miniconda3/envs/fom/bin/python
# python pipeline.py --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 1.0 --z 0.1 --repo test --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 3 --repo test
import os
print("PID:",os.getpid())

import logging
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from few.utils.geodesic import get_separatrix
from few.trajectory.ode import KerrEccEqFlux
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import inner_product
from common import CosmoInterpolator
import time
import matplotlib.pyplot as plt
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot
from waveform_utils import initialize_waveform_generator, transf_log_e_wave, generate_random_phases, generate_random_sky_localization
from few.utils.geodesic import get_fundamental_frequencies
from scipy.signal.windows import tukey
#psd stuff
from psd_utils import load_psd, get_psd_kwargs

cosmo = CosmoInterpolator()

# Initialize logger
logger = logging.getLogger()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--M", help="Primary Mass of the central black hole at detector", type=float)
    parser.add_argument("--mu", help="Secondary Mass of the compact object at detector", type=float)
    parser.add_argument("--a", help="Dimensionless Spin of the central black hole", type=float)
    parser.add_argument("--e_f", help="Final eccentricity at separatrix + 0.1", type=float)
    parser.add_argument("--T", help="Observation time", type=float)
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
    parser.add_argument('--model', type=str, default="scirdv1", help="Noise model to use") #TODO: is this used?
    parser.add_argument("--calculate_fisher", help="Calculate the Fisher matrix", type=int, default=0)
    # optional time to plunge
    parser.add_argument("--Tpl", help="Time to plunge", type=float, default=0.0)
    
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


from scipy.signal import get_window
from matplotlib.colors import LogNorm

class wave_windowed_truncated():
    def __init__(self, wave_gen, N, dt, xp, window_fn=('tukey', 0.005), fmin=0.0, fmax=1.0):
        self.wave_gen = wave_gen
        self.window_fn = window_fn
        self.window = xp.asarray(get_window(self.window_fn, N))
        self.frequency = xp.fft.rfftfreq(N, dt)
        self.mask = (self.frequency > fmin) * (self.frequency < fmax)
        self.xp = xp
        self.N = N
    
    def __call__(self, *args, **kwargs):
        wave = xp.asarray(self.wave_gen(*args, **kwargs))
        # apply window
        wave = wave * self.window
        # take fft
        wave_fft = self.xp.fft.rfft(wave,axis=1)
        wave_fft[:,~self.mask] = 0.0 + 1j*0.0
        # take ifft
        wave = self.xp.fft.irfft(wave_fft,axis=1, n=self.N)
        return wave

    def __getattr__(self, name):
        # Forward attribute access to base_wave
        return getattr(self.wave_gen, name)


inspiral_kwargs_back = {"err": 1e-10,"integrate_backwards": True}
inspiral_kwargs_forward = {"err": 1e-10,"integrate_backwards": False}

param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])
popinds = []
popinds.append(5)
popinds.append(12)
param_names = np.delete(param_names, popinds).tolist()

if __name__ == "__main__":
    start_script = time.time()
    
    args = parse_arguments()
    #args = process_args(args)
    xp = initialize_gpu(args)

    # create repository
    os.makedirs(args.repo, exist_ok=True)

    # PSD
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
    M = args.M
    mu = args.mu
    a = np.abs(args.a)
    e_f = args.e_f
    x0_f = 1.0 * np.sign(args.a) if args.a != 0.0 else 1.0
    p_f = get_separatrix(a, args.e_f, x0_f) + 0.1
    dist = cosmo.get_luminosity_distance(args.z)
    print("Distance in Gpc", dist)
    # observation time
    if args.Tpl > 0.0:
        Tpl = args.Tpl
    else:
        Tpl = args.T
    
    T = args.T

    # initialize the trajectory
    traj = EMRIInspiral(func=KerrEccEqFlux)
    print("Generating backward trajectory")
    t_forward, p_forward, e_forward, x_forward, Phi_phi_forward, Phi_r_forward, Phi_theta_forward = traj(M, mu, a, p_f, e_f, x0_f, dt=1e-5, T=100.0, integrate_backwards=False)
    t_back, p_back, e_back, x_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(M, mu, a, p_forward[-1], e_forward[-1], x_forward[-1], dt=1e-5, T=Tpl, integrate_backwards=True)
    # and forward trajectory to check the evolution
    t_plot, p_plot, e_plot, x_plot, Phi_phi_plot, Phi_r_plot, Phi_theta_plot = traj(M, mu, a, p_back[-1], e_back[-1], x_back[-1], dt=1e-5, T=T, integrate_backwards=False)
    # Plot (t, p), (t, e), (p, e), (t, Phi_phi)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # (t, p)
    axs[0, 0].plot(t_plot, p_plot)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("p")
    axs[0, 0].set_title("(t, p)")

    # (t, e)
    axs[0, 1].plot(t_plot, e_plot)
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("e")
    axs[0, 1].set_title("(t, e)")

    # (p, e)
    axs[1, 0].plot(p_plot, e_plot)
    axs[1, 0].set_xlabel("p")
    axs[1, 0].set_ylabel("e")
    axs[1, 0].set_title("(p, e)")

    # (t, Phi_phi)
    axs[1, 1].plot(t_plot, Phi_phi_plot)
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Phi_phi")
    axs[1, 1].set_title("(t, Phi_phi)")

    plt.tight_layout()
    plt.savefig(os.path.join(args.repo, "trajectory_subplots.png"))
    plt.close(fig)
    print("Found initial conditions", p_back[-1], e_back[-1], x_back[-1])
    omegaPhi, omegaTheta, omegaR = get_fundamental_frequencies(a, p_back, e_back, x_back)
    dimension_factor = 2.0 * np.pi * M * MTSUN_SI
    omegaPhi = omegaPhi / dimension_factor
    omegaTheta = omegaTheta / dimension_factor
    omegaR = omegaR / dimension_factor
    print("Done with the trajectory")
    # define the initial parameters
    p0, e0, x0 = p_back[-1], e_back[-1], x_back[-1]
    print("p0, e0, x0", p0, e0, x0)
    # initialiaze the waveform generator
    temp_model = initialize_waveform_generator(T, args, inspiral_kwargs_forward)
    # base waveform has always the same parameters for comparison
    Phi_phi0, Phi_r0, Phi_theta0 = 0.0, 0.0, 0.0# generate_random_phases()
    qS, phiS, qK, phiK = np.pi/3, np.pi/3, np.pi/3, np.pi/3 # generate_random_sky_localization()
    parameters = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])
    # evaluate waveform
    temp_model(*parameters, mode_selection_threshold=1e-5)

    tic = time.time()
    waveform_out = temp_model(*parameters, mode_selection_threshold=1e-5)
    toc = time.time()
    timing = toc - tic
    print("Time taken for one waveform generation: ", timing)
    print("\n")
    # define frequency ranges for inner product
    ns = temp_model.waveform_gen.waveform_generator.ns
    ms = temp_model.waveform_gen.waveform_generator.ms
    ls = temp_model.waveform_gen.waveform_generator.ls
    max_f = float(np.max(np.abs(omegaPhi[None,:] * ms.get()[:,None] + omegaR[None,:] * ns.get()[:,None]))) * 1.01 # added a 1% safety factor
    # define modes for waveform
    waveform_kwargs = {"mode_selection": [(ll,mm,nn) for ll,mm,nn in zip(ls.get(), ms.get(), ns.get())],}
    # create a waveform that is windowed and truncated 
    test_1 = np.sum(np.abs(temp_model(*parameters, **waveform_kwargs)[0] - temp_model(*parameters, mode_selection=[(2,2,0)])[0]))
    test_2 = np.sum(np.abs(temp_model(*parameters, **waveform_kwargs)[0] - waveform_out[0]))
    print("Test 1: ", test_1 !=0.0, "\nTest 2: ", test_2 == 0.0)
    # update the model with the windowed and truncated waveform
    model = wave_windowed_truncated(temp_model, len(waveform_out[0]), args.dt, xp, window_fn=('tukey', 0.01), fmin=1e-5, fmax=max_f)
    model(*parameters)
    tic = time.time()
    waveform_out = model(*parameters)
    toc = time.time()
    timing = toc - tic
    print("Time taken for one waveform generation: ", timing)
    # save the waveform generation time
    with open(os.path.join(args.repo, "waveform_generation_time.txt"), "w") as f:
        f.write(str(timing))
    # check if there are nans in the waveform_out[0]
    if xp.isnan(xp.asarray(waveform_out)).any():
        print("There are nans in the waveform")

    fft_waveform = xp.fft.rfft(waveform_out[0]).get() * args.dt
    freqs = np.fft.rfftfreq(len(waveform_out[0]), d=args.dt)
    mask = (freqs>1e-5)
    plt.figure()
    plt.loglog(freqs[mask], np.abs(fft_waveform)[mask]**2)
    plt.loglog(freqs[mask], np.atleast_2d(psd_wrap(freqs[mask]).get())[0], label="PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude $|\tilde h(f)|$")
    plt.legend()
    plt.ylim(1e-45, 1e-35)
    plt.savefig(os.path.join(args.repo, "waveform_frequency_domain.png"))
    # Plot waveform in time domain
    plt.figure()
    plt.plot(np.arange(len(waveform_out[0].get())) * args.dt, waveform_out[0].get(), label="A")
    plt.plot(np.arange(len(waveform_out[0].get())) * args.dt, waveform_out[1].get(), label="E", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.title("Waveform in Time Domain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.repo, "waveform_time_domain.png"))
    # Plot spectrogram
    plt.figure()
    plt.specgram(waveform_out[0].get(), NFFT=int(86400/args.dt), Fs=1/args.dt, noverlap=128, scale='dB', cmap='viridis')
    plt.yscale('log')
    plt.ylim(1e-5, 1e-1)  # Adjust y-axis limits for better visibility
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram (Real part)")
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(os.path.join(args.repo, "waveform_spectrogram.png"))
    # plt.close("all")
    # check horizon d_L
    # d_L = inner_product(waveform_out, waveform_out, psd_wrap(freqs[1:]), dt=args.dt, use_gpu=args.use_gpu)**0.5/20.
    # redshift = get_redshift(d_L)
    # source_frame_m1 = parameters[0] / (1 + redshift)
    # source_frame_m2 = parameters[1] / (1 + redshift)
    # plt.figure(); plt.loglog(redshift, d_L); plt.xlabel("Redshift"); plt.grid(); plt.savefig(os.path.join(args.repo, "snr_vs_redshift.png"))
    # if low eccentricity, use the log_e transformation
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
        # update the parameters
        parameters = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])

        current_folder = os.path.join(args.repo, name_realization)
        # create folder for the realization
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
                                # window=window # addition of the window to avoid leakage
                                waveform_kwargs=waveform_kwargs
                                )
        # calculate the SNR
        SNR = fish.SNRcalc_SEF()
        np.savez(os.path.join(current_folder, "snr.npz"), snr=SNR, parameters=parameters, redshift=args.z, e_f=args.e_f, Tplunge=T)
        
        calculate_fisher = bool(args.calculate_fisher)
        if calculate_fisher:
            # calculate the Fisher matrix
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
            
            J = cosmo.jacobian(M / (1 + args.z), mu / (1 + args.z), args.z)
            source_frame_cov = J @ cov @ J.T

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
            Sigma = cov[6:8, 6:8]
            err_sky_loc = 2 * np.pi * np.sin(qS) * np.sqrt(np.linalg.det(Sigma)) * (180.0 / np.pi) ** 2
            np.savez(os.path.join(current_folder, "results.npz"), gamma=fim, cov=cov, snr=SNR, fisher_params=fisher_params, errors=errors, relative_errors=errors/fisher_params, names=param_names, source_frame_cov=source_frame_cov, err_sky_loc=err_sky_loc, redshift=args.z, e_f=args.e_f)
            print("Saved results to", current_folder)
            print("*************************************")
            


    
    end_script = time.time()
    print("Total time taken for the script: ", end_script - start_script)
    # save total time taken
    with open(os.path.join(args.repo, "total_time_script.txt"), "w") as f:
        f.write(str(end_script - start_script))