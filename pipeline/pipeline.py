# python pipeline.py --M 1e6 --mu 1e1 --a 0.5 --e_f 0.1 --T 4.0 --z 0.5 --psd_file TDI2_AE_psd.npy --dt 10.0 --use_gpu --N_montecarlo 1 --device 0 --power_law --repo test_acc --calculate_fisher 1
# singularity exec --nv ../fom_final.sif python pipeline.py --M 50000.0 --mu 50.0 --a 0.5 --e_f 0.0 --T 0.25 --z 0.5 --psd_file TDI2_AE_psd.npy --dt 0.6 --use_gpu --N_montecarlo 1 --device 0 --repo test_ --calculate_fisher 1
import os
print("PID:",os.getpid())

import logging
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
#from few.utils.geodesic import get_separatrix
from few.trajectory.ode import KerrEccEqFlux
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import inner_product
from common import CosmoInterpolator
import time
import matplotlib.pyplot as plt
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot
from waveform_utils import initialize_waveform_generator, transf_log_e_wave, generate_random_phases, generate_random_sky_localization, wave_windowed_truncated
from few.utils.geodesic import get_fundamental_frequencies
from scipy.signal.windows import tukey
#psd stuff
from psd_utils import load_psd, get_psd_kwargs
import h5py

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
    parser.add_argument("--calculate_fisher", help="Calculate the Fisher matrix", type=int, default=0)
    # optional time to plunge
    parser.add_argument("--Tpl", help="Time to plunge", type=float, default=0.0)
    # arguments 
    parser.add_argument('--power_law', action='store_true', default=False, help="Consider beyond-vacuum GR power-law correction")
    parser.add_argument("--nr", help="power-law", type=float, default=8.0)
    
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
    

class KerrEccEqFluxPowerLaw(KerrEccEqFlux):
    def modify_rhs(self, ydot, y):
        # in-place modification of the derivatives
        LdotAcc = (
            self.additional_args[0]
            * pow(y[0] / 10.0, self.additional_args[1])
            * 32.0
            / 5.0
            * pow(y[0], -7.0 / 2.0)
        )
        dL_dp = (
            -3 * pow(a, 3)
            + pow(a, 2) * (8 - 3 * y[0]) * np.sqrt(y[0])
            + (-6 + y[0]) * pow(y[0], 2.5)
            + 3 * a * y[0] * (-2 + 3 * y[0])
        ) / (2.0 * pow(2 * a + (-3 + y[0]) * np.sqrt(y[0]), 1.5) * pow(y[0], 1.75))
        # transform back to pdot from Ldot and add GW contribution
        ydot[0] = ydot[0] + LdotAcc / dL_dp


        


if __name__ == "__main__":
    start_script = time.time()
    
    args = parse_arguments()
    #args = process_args(args)
    xp = initialize_gpu(args)

    # create repository
    os.makedirs(args.repo, exist_ok=True)

    A = 0
    nr = args.nr
    KerrEccEqFluxPowerLaw().additional_args = (A, nr)  # A is the power-law coefficient, nr is the power-law exponent

    inspiral_kwargs_back = {"err": 1e-13,"integrate_backwards": True, "func":  KerrEccEqFluxPowerLaw}
    inspiral_kwargs_forward = {"err": 1e-13,"integrate_backwards": False, "func":  KerrEccEqFluxPowerLaw}     

    param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0', 'A', 'nr'])
    
    popinds = []
    popinds.append(5)
    popinds.append(12)
    
    if args.power_law:
        popinds.append(4)
        popinds.append(13)
        popinds.append(15)
    
    else:
        popinds.append(14)
        popinds.append(15)
        if args.e_f == 0.0:
            popinds.append(4)
            popinds.append(13)
        
    param_names = np.delete(param_names, popinds).tolist()
    
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
    if args.power_law: #Beyond-GR only considered for circular case
        e_f = 0.0
    else:
        e_f = args.e_f
        inspiral_kwargs_forward["err"] = 1e-10
    
    x0_f = 1.0 * np.sign(args.a) if args.a != 0.0 else 1.0
    
    traj = EMRIInspiral(func=KerrEccEqFluxPowerLaw)
    # if args.power_law:
    #     p_f = KerrEccEqFluxPowerLaw().min_p(e=e_f, x=x0_f, a=a) + 0.5
    # else:
    #     # p_f = KerrEccEqFluxPowerLaw().min_p(e=e_f, x=x0_f, a=a) + 0.1
    p_f = traj.func.min_p(e_f, x0_f, a)
    dist = cosmo.get_luminosity_distance(args.z)
    print("Distance in Gpc", dist)
    # observation time
    if args.Tpl > 0.0:
        Tpl = args.Tpl
    else:
        Tpl = args.T
    
    T = args.T

    # initialize the trajectory
    print("Generating backward trajectory")
    t_forward, p_forward, e_forward, x_forward, Phi_phi_forward, Phi_r_forward, Phi_theta_forward = traj(M, mu, a, p_f, e_f, x0_f, A, nr, dt=1e-5, T=100.0, integrate_backwards=False)
    t_back, p_back, e_back, x_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(M, mu, a, p_forward[-1], e_forward[-1], x_forward[-1], A, nr, dt=1e-5, T=Tpl, integrate_backwards=True)
    # and forward trajectory to check the evolution
    t_plot, p_plot, e_plot, x_plot, Phi_phi_plot, Phi_r_plot, Phi_theta_plot = traj(M, mu, a, p_back[-1], e_back[-1], x_back[-1], A, nr, dt=1e-5, T=T, integrate_backwards=False)
    # save information about the trajectory in h5 file
    with h5py.File(os.path.join(args.repo, "trajectory.h5"), "w") as f:
        f.create_dataset("t_plot", data=t_plot)
        f.create_dataset("p_plot", data=p_plot)
        f.create_dataset("e_plot", data=e_plot)
        f.create_dataset("x_plot", data=x_plot)
        f.create_dataset("Phi_phi_plot", data=Phi_phi_plot)
        f.create_dataset("Phi_r_plot", data=Phi_r_plot)
        f.create_dataset("Phi_theta_plot", data=Phi_theta_plot)
    
    T = t_plot[-1]/YRSID_SI
    print("Total observation time in years:", T)
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
    plt.close('all')
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
    print(phiK,phiS)
    parameters = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, A, nr])
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
    min_f = float(np.min(np.abs(omegaPhi[None,:] * ms.get()[:,None] + omegaR[None,:] * ns.get()[:,None]))) * 0.99 # added a 1% safety factor
    # define modes for waveform
    waveform_kwargs = {"mode_selection": [(ll,mm,nn) for ll,mm,nn in zip(ls.get(), ms.get(), ns.get())],}
    print("Number of modes in the waveform:", len(waveform_kwargs["mode_selection"]))
    # create a waveform that is windowed and truncated 
    test_1 = np.sum(np.abs(temp_model(*parameters, **waveform_kwargs)[0] - temp_model(*parameters, mode_selection=[(2,2,0)])[0]))
    test_2 = np.sum(np.abs(temp_model(*parameters, **waveform_kwargs)[0] - waveform_out[0]))
    print("Test 1: ", test_1 !=0.0, "\nTest 2: ", test_2 == 0.0)
    # update the model with the windowed and truncated waveform
    fmin = np.max([0.5e-4, min_f])
    fmax = np.min([1.0, 1/(args.dt*2), max_f])
    model = wave_windowed_truncated(temp_model, xp, t0=100000.0)
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
    mask = (freqs>fmin) & (freqs<fmax)
    plt.figure()
    
    plt.loglog(freqs[mask], np.abs(fft_waveform)[mask]**2 / (len(waveform_out[0]) * args.dt), label="Waveform")
    plt.loglog(freqs[mask], np.atleast_2d(psd_wrap(freqs[mask]).get())[0], label="PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Amplitude $|\tilde h(f)| df$")
    plt.legend()
    plt.ylim(1e-45, 1e-32)
    plt.savefig(os.path.join(args.repo, "waveform_frequency_domain.png"))
    
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
    plt.ylim(5e-5, 0.0)  # Adjust y-axis limits for better visibility
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram (Real part)")
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(os.path.join(args.repo, "waveform_spectrogram.png"))
    plt.close("all")
    
    # check horizon d_L
    # d_L = inner_product(waveform_out, waveform_out, psd_wrap(freqs[1:]), dt=args.dt, use_gpu=args.use_gpu)**0.5/20.
    # redshift = get_redshift(d_L)
    # source_frame_m1 = parameters[0] / (1 + redshift)
    # source_frame_m2 = parameters[1] / (1 + redshift)
    # plt.figure(); plt.loglog(redshift, d_L); plt.xlabel("Redshift"); plt.grid(); plt.savefig(os.path.join(args.repo, "snr_vs_redshift.png"))
    # if low eccentricity, use the log_e transformation
    if (args.e_f < 1e-3) and (args.e_f != 0.0):
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
        print(phiK,phiS)
        # update the parameters
        parameters = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, A, nr])

        current_folder = os.path.join(args.repo, name_realization)
        # create folder for the realization
        os.makedirs(current_folder, exist_ok=True)
        # save the parameters to txt file
        np.savetxt(os.path.join(current_folder, "all_parameters.txt"), parameters.T, header=" ".join(param_names))

        if args.power_law:
            der_order = 8.0
        else:
            der_order = 4.0


        fish = StableEMRIFisher(*parameters[:-2], 
                                add_param_args = {"A": A, "nr": nr}, 
                                fmin=fmin, fmax=fmax,
                                dt=args.dt, T=T, 
                                EMRI_waveform_gen=EMRI_waveform_gen, 
                                noise_model=psd_wrap, 
                                noise_kwargs=dict(TDI="TDI2"), 
                                channels=["A", "E"], 
                                param_names=param_names, 
                                stats_for_nerds=False, use_gpu=args.use_gpu, 
                                der_order=der_order, Ndelta=20, filename=current_folder,
                                deltas = deltas,
                                log_e = log_e, # useful for sources close to zero eccentricity
                                CovEllipse=False, # will return the covariance and plot it
                                stability_plot=False, # activate if unsure about the stability of the deltas
                                # window=window # addition of the window to avoid leakage
                                waveform_kwargs=waveform_kwargs
                                )
        # calculate the SNR
        SNR = fish.SNRcalc_SEF()
        
        accumulation_index = np.arange(len(waveform_out[0])//20,len(waveform_out[0]),len(waveform_out[0])//20, dtype=int)

        np.savez(os.path.join(current_folder, "snr.npz"), snr=SNR, parameters=parameters, redshift=args.z, e_f=args.e_f, Tplunge=T)
        
        calculate_fisher = bool(args.calculate_fisher)
        if args.calculate_fisher:
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
             
            if args.power_law:
                J = cosmo.jacobian_powerlaw(M / (1 + args.z), mu / (1 + args.z), args.z)
            else:
                J = cosmo.jacobian(M / (1 + args.z), mu / (1 + args.z), args.z)
            source_frame_cov = J @ cov @ J.T

            if j == 0:
                CovEllipsePlot(fish.param_names, fish.wave_params, cov, filename=current_folder + f"/covariance_ellipse_plot.png")
            
            # get errors
            errors = np.sqrt(np.diag(cov))
            fisher_params = np.delete(parameters, popinds)
            # Use absolute errors when parameter value is zero
            relative_errors = np.zeros_like(errors)
            for i, param in enumerate(fisher_params):
                if param == 0 or np.isclose(param, 0):
                    relative_errors[i] = errors[i] 
                else:
                    relative_errors[i] = errors[i] / param
            
            errors_df = {"Parameter": param_names, "parameter value": fisher_params, "1 sigma Error": errors, "Relative Error": relative_errors, "SNR": SNR}
            errors_df = pd.DataFrame(errors_df)
            errors_df.to_markdown(os.path.join(current_folder, "summary.md"), floatfmt=".10e")
            # save the covariance matrix and the SNR to npz file
            Sigma = cov[6:8, 6:8]
            err_sky_loc = 2 * np.pi * np.sin(qS) * np.sqrt(np.linalg.det(Sigma)) * (180.0 / np.pi) ** 2
            np.savez(os.path.join(current_folder, "results.npz"), gamma=fim, cov=cov, snr=SNR, fisher_params=fisher_params, errors=errors, relative_errors=relative_errors, names=param_names, source_frame_cov=source_frame_cov, err_sky_loc=err_sky_loc, redshift=args.z, e_f=args.e_f)
            print("Saved results to", current_folder)
            print("*************************************")
            

    # Plot waveform in time domain
    N = len(waveform_out[0])
    accumulation_index = np.arange(N//20,len(waveform_out[0]), N//20, dtype=int)
    accumulation_time = accumulation_index * args.dt
    snr_accumation = [inner_product(waveform_out[:,:ii],waveform_out[:,:ii], psd_wrap(np.fft.rfftfreq(len(waveform_out[0][:ii]), d=args.dt)[1:]), args.dt, fmin = fmin, fmax = fmax, use_gpu=args.use_gpu)**0.5for ii in accumulation_index]
    snr_accumation = np.array(snr_accumation)
    
    plt.figure()
    plt.plot(accumulation_time, snr_accumation, 'o')
    plt.xlabel('t')
    plt.ylabel('SNR(t)')
    plt.title('Accumulated SNR over time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.repo, "accumulated_snr.png"))
    plt.close("all")
    # ============================================
    # POSTPROCESSING: Aggregate all realizations
    # ============================================
    print("\n" + "="*50)
    print("POSTPROCESSING: Aggregating all realizations")
    print("="*50)
    
    import glob
    import h5py
    
    # Create HDF5 file for aggregated results
    h5_filename = os.path.join(args.repo, "aggregated_results.h5")
    with h5py.File(h5_filename, "w") as h5f:
        # Get list of realization folders
        realization_folders = sorted([d for d in glob.glob(os.path.join(args.repo, "realization_*")) if os.path.isdir(d)])
        print(f"Found {len(realization_folders)} realizations to process")
        
        # Load SNR data
        snr_list = np.asarray([np.load(el)["snr"] for el in sorted(glob.glob(os.path.join(args.repo, "*/snr.npz")))])
        detector_params = np.asarray([np.load(el)["parameters"] for el in sorted(glob.glob(os.path.join(args.repo, "*/snr.npz")))])
        
        redshift = np.load(sorted(glob.glob(os.path.join(args.repo, "*/snr.npz")))[0])["redshift"]
        e_f_val = np.load(sorted(glob.glob(os.path.join(args.repo, "*/snr.npz")))[0])["e_f"]
        Tpl_val = np.load(sorted(glob.glob(os.path.join(args.repo, "*/snr.npz")))[0])["Tplunge"]
        
        # Prepare source frame parameters
        source_params = detector_params[0].copy()
        source_params[0] = source_params[0] / (1 + redshift)
        source_params[1] = source_params[1] / (1 + redshift)
        
        # Extract parameters
        lum_dist = detector_params[:, 6]
        sky_loc = detector_params[:, 7:9]
        spin_loc = detector_params[:, 9:11]
        detector_params_ref = detector_params[0]
        
        # Prepare result dictionary
        result = {
            "m1": source_params[0],
            "m2": source_params[1],
            "a": source_params[2] * source_params[5],
            "p0": source_params[3],
            "e0": source_params[4],
            "DL": source_params[6],
            "e_f": e_f_val,
            "Tpl": Tpl_val,
            "redshift": redshift,
            "lum_dist": lum_dist,
            "snr": snr_list,
            "sky_loc": sky_loc,
            "spin_loc": spin_loc,
            "accumulation_time": accumulation_time,
            "snr_accumation": snr_accumation,
        }
        
        # Store SNR results in HDF5
        grp = h5f.create_group("SNR_analysis")
        for k, v in result.items():
            grp.create_dataset(k, data=v)
        
        # SNR histogram
        plt.figure(figsize=(8, 6))
        plt.hist(snr_list, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('SNR')
        plt.ylabel('Counts')
        plt.title(f'SNR Distribution (N={len(snr_list)})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.repo, "snr_histogram.png"), dpi=300)
        plt.close()
        print("Saved SNR histogram")
        
        # Check if Fisher matrix results exist
        fisher_results = sorted(glob.glob(os.path.join(args.repo, "*/results.npz")))
        
        if len(fisher_results) > 0 and args.calculate_fisher:
            print(f"\nProcessing Fisher matrix results ({len(fisher_results)} realizations)")
            
            # Load Fisher matrix data
            source_cov = np.asarray([np.load(el)["source_frame_cov"] for el in fisher_results])
            detector_cov = np.asarray([np.load(el)["cov"] for el in fisher_results])
            fish_params = np.asarray([np.load(el)["fisher_params"] for el in fisher_results])
            err_sky_loc = np.asarray([np.load(el)["err_sky_loc"] for el in fisher_results])
            names = np.asarray([np.load(el)["names"] for el in fisher_results])[0]
            
            # Transform to source frame for masses
            fish_params[:, 0] = fish_params[:, 0] / (1 + redshift)
            fish_params[:, 1] = fish_params[:, 1] / (1 + redshift)
            
            # Calculate measurement precision
            measurement_precision = np.asarray([np.sqrt(np.diag(source_cov[ii])) for ii in range(len(fish_params))])
            detector_measurement_precision = np.asarray([np.sqrt(np.diag(detector_cov[ii])) for ii in range(len(fish_params))])
            
            # Prepare parameter names for Fisher analysis
            fisher_names = names.copy()
            fisher_names[6] = "Omega"  # Sky location
            fisher_names[7] = "iota"   # Inclination
            fisher_names = fisher_names[:8]  # Keep only up to inclination
            
            # Update sky location and inclination errors
            measurement_precision[:, 6] = err_sky_loc
            detector_measurement_precision[:, 6] = err_sky_loc
            measurement_precision[:, 7] = measurement_precision[:, 8]
            detector_measurement_precision[:, 7] = detector_measurement_precision[:, 8]
            
            # Create Fisher analysis group in HDF5
            fisher_grp = h5f.create_group("Fisher_analysis")
            fisher_grp.create_dataset("fisher_params", data=fish_params)
            fisher_grp.create_dataset("measurement_precision_source", data=measurement_precision)
            fisher_grp.create_dataset("measurement_precision_detector", data=detector_measurement_precision)
            fisher_grp.create_dataset("param_names", data=np.asarray(fisher_names, dtype=h5py.string_dtype(encoding='utf-8')))
            
            # Generate plots for each parameter
            for ii, param_name in enumerate(fisher_names):
                error_source = measurement_precision[:, ii]
                error_detector = detector_measurement_precision[:, ii]
                
                # Determine if relative or absolute error
                if (param_name == "M") or (param_name == "mu"):
                    error_source_plot = error_source / fish_params[:, ii]
                    error_detector_plot = error_detector / detector_params_ref[ii]
                    xlabel = f'Relative error {param_name}'
                    group_name = f"relative_errors_{param_name}"
                else:
                    error_source_plot = error_source
                    error_detector_plot = error_detector
                    xlabel = f'Absolute error {param_name}'
                    group_name = f"absolute_errors_{param_name}"
                
                # Create parameter group
                param_grp = fisher_grp.create_group(group_name)
                param_grp.create_dataset("error_source", data=error_source_plot)
                param_grp.create_dataset("error_detector", data=error_detector_plot)
                
                # Histogram plot
                plt.figure(figsize=(8, 6))
                plt.hist(error_source_plot, bins=30, alpha=0.6, label='Source frame', edgecolor='black')
                plt.hist(error_detector_plot, bins=30, alpha=0.6, label='Detector frame', edgecolor='black')
                plt.xlabel(xlabel)
                plt.ylabel('Counts')
                plt.title(f'Error Distribution: {param_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(args.repo, f"{param_name}_histogram.png"), dpi=300)
                plt.close()
                
                # SNR vs error plot
                plt.figure(figsize=(8, 6))
                plt.plot(snr_list, error_source_plot, 'o', label='Source frame', alpha=0.6)
                plt.plot(snr_list, error_detector_plot, 'x', label='Detector frame', alpha=0.6)
                
                # Add scaling reference lines
                snr_vec = np.linspace(np.min(snr_list), np.max(snr_list), len(lum_dist))
                if param_name == "DL":
                    plt.plot(snr_vec, lum_dist / snr_vec, 'r--', label='d/SNR')
                else:
                    plt.plot(snr_vec, 1 / snr_vec, 'r--', label='1/SNR')
                
                plt.xlabel('SNR')
                plt.ylabel(xlabel)
                plt.title(f'SNR vs Error: {param_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xscale('log')
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(os.path.join(args.repo, f"snr_{param_name}.png"), dpi=300)
                plt.close()
            
            print(f"Saved Fisher matrix analysis plots for {len(fisher_names)} parameters")
        else:
            print("No Fisher matrix results found or calculate_fisher=False. Skipping Fisher analysis.")
    
    print(f"\nAggregated results saved to {h5_filename}")
    
    end_script = time.time()
    print("Total time taken for the script: ", end_script - start_script)
    # save total time taken
    with open(os.path.join(args.repo, "total_time_script.txt"), "w") as f:
        f.write(str(end_script - start_script))