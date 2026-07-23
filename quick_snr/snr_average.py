import argparse
import multiprocess
import warnings
import time

import h5py
import numpy as np
from astropy.cosmology import Planck18 as COSMO
from scipy.interpolate import CubicSpline

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux, get_separatrix
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import YRSID_SI

warnings.filterwarnings("ignore")

DT = 10.0
EPS = 1e-2
MODES = [(ll, mm, nn) for ll in range(2, 5) for mm in range(1, ll + 1) for nn in range(-1, 3)]

DEFAULT_ANGLES = {
    "qS": np.pi / 3,
    "phiS": np.pi / 3,
    "qK": np.pi / 3,
    "phiK": np.pi / 3,
    "Phi_phi0": np.pi / 3,
    "Phi_theta0": 0.0,
    "Phi_r0": np.pi / 3,
}


global FEW_GEN, TRAJ, RHS

TRAJ = EMRIInspiral(func=KerrEccEqFlux)
RHS = KerrEccEqFlux()
FEW_GEN = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    return_list=True,
)


def redshift_to_luminosity_distance(z):
    return COSMO.luminosity_distance(z).value * 1e-3  # Gpc


def get_spline_psd(filename="LISA v1.0_PSD.h5"):
    with h5py.File(filename, "r") as data:
        key = filename.split("_PSD.h5")[0]
        psd_data = data[key]["sensitivities_links"][()]
        f_psd = data[key]["f"][()]
    return CubicSpline(f_psd, psd_data)


def get_spline_psd_alice(filename):
    f_psd, asd_psd = np.loadtxt(filename, unpack=True)
    return CubicSpline(f_psd, asd_psd ** 2)


def get_psd_wrapper(psd="LISA"):
    if psd == "LISA":
        f_psd, asd_psd = np.loadtxt("PSD_plus_foregrounds_LISA_LS_asd.dat", unpack=True)
        cubic_spline_psd = CubicSpline(f_psd, asd_psd**2)
        # print(f"Using PSD from PSD_plus_foregrounds_LISA_LS_asd.dat...", cubic_spline_psd)
        fmin, fmax = 1e-4, 1.0
    elif psd == "AMADEUS":
        f_psd, asd_psd = np.loadtxt("PSD_plus_foreground_AMADEUS-Baseline_asd.txt", unpack=True)
        cubic_spline_psd = CubicSpline(f_psd, asd_psd**2)
        # print(f"Using PSD from PSD_plus_foreground_AMADEUS-Baseline_asd.txt...", cubic_spline_psd)
        fmin, fmax = 1e-6, 1.0
    elif psd == "DO-IT":
        f_psd, asd_psd = np.loadtxt("PSD_plus_foreground_DO-IT-Baseline_asd.txt", unpack=True)
        cubic_spline_psd = CubicSpline(f_psd, asd_psd**2)
        # print(f"Using PSD from PSD_plus_foreground_DO-IT-Baseline_asd.txt...", cubic_spline_psd)
        fmin, fmax = 1e-4, 10.0
    
    return cubic_spline_psd, fmin, fmax


def get_initial_conditions(params, Tobs=None, err=1e-6):
    m1, m2, a, Tpl, ef = params
    x0 = 1.0
    RHS.add_fixed_parameters(m1, m2, a)

    p_0 = TRAJ.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, ef, x0) + 1e-3
    forward_result = TRAJ(m1, m2, a, p_0, ef, x0, T=1e6, integrate_backwards=False, err=err)
    backwards_result = TRAJ(
        m1,
        m2,
        a,
        forward_result[1][-1],
        forward_result[2][-1],
        x0,
        T=Tpl,
        integrate_backwards=True,
        err=err,
    )

    p0 = backwards_result[1][-1]
    e0 = backwards_result[2][-1]
    x0 = backwards_result[3][-1]
    # check that the final time is close to Tpl
    if np.abs(1-backwards_result[0][-1]/YRSID_SI / Tpl) > 1e-3:
        print(f"Final time {backwards_result[0][-1]/YRSID_SI} is not close to Tpl={Tpl}")
    
    if Tobs is not None:
        forward_result = TRAJ(m1, m2, a, p0, e0, x0, T=Tobs, integrate_backwards=False, err=err)
    
    f_phi_theta_r = TRAJ.inspiral_generator.eval_integrator_derivative_spline(forward_result[0], order=1)
    f_phi = f_phi_theta_r[:, 3] / (2 * np.pi)
    f_r = f_phi_theta_r[:, 5] / (2 * np.pi)
    return p0, e0, x0, f_phi, f_r


def compute_snr(
    m1,
    m2,
    a,
    Tpl,
    ef,
    z,
    Tobs=None,
    qS=None,
    phiS=None,
    qK=None,
    phiK=None,
    Phi_phi0=None,
    Phi_theta0=None,
    Phi_r0=None,
    psd="LISA",
    num_freq=5000,
):

    qS = DEFAULT_ANGLES["qS"] if qS is None else qS
    phiS = DEFAULT_ANGLES["phiS"] if phiS is None else phiS
    qK = DEFAULT_ANGLES["qK"] if qK is None else qK
    phiK = DEFAULT_ANGLES["phiK"] if phiK is None else phiK
    Phi_phi0 = DEFAULT_ANGLES["Phi_phi0"] if Phi_phi0 is None else Phi_phi0
    Phi_theta0 = DEFAULT_ANGLES["Phi_theta0"] if Phi_theta0 is None else Phi_theta0
    Phi_r0 = DEFAULT_ANGLES["Phi_r0"] if Phi_r0 is None else Phi_r0
    if Tobs is None:
        Tobs = Tpl
    
    dist = redshift_to_luminosity_distance(z)
    try:
        p0, e0, x0, fphi, fr = get_initial_conditions(np.asarray([m1 * (1 + z), m2 * (1 + z), a, Tpl, ef]), Tobs=Tobs)
        # print(f"Initial conditions for m1={m1}, m2={m2}, a={a}, Tpl={Tpl}, ef={ef}, z={z}: p0={p0}, e0={e0}")
        print(f"Frequency range 2 fphi: {2*fphi.min()} - {2*fphi.max()} Hz")
    except Exception as exc:
        print(f"Error computing initial conditions for m1={m1}, m2={m2}, a={a}, Tpl={Tpl}, ef={ef}, z={z}: {exc}")
        return 0.0

    cubic_spline_psd, fmin, fmax = get_psd_wrapper(psd)
    fmax = min(fmax, 5*fphi.max())  # Limit fmax to 4 times the last orbital frequency to avoid extrapolation
    fmin = max(fmin, 1*fphi.min())  # Limit fmin to 0.1 times the first orbital frequency to avoid extrapolation
    f_pos = np.linspace(fmin, fmax, num=num_freq)
    freq = np.hstack((-f_pos[::-1], np.asarray([0.0]), f_pos))
    # print(f"fmin={fmin}, fmax={fmax}")

    hf = FEW_GEN(
        m1 * (1 + z),
        m2 * (1 + z),
        a,
        p0,
        e0,
        x0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        T=Tobs,
        dt=DT,
        f_arr=freq,
        mask_positive=True,
        mode_selection=MODES,
    )

    h_plus = np.asarray(hf[0])[1:]
    h_cross = np.asarray(hf[1])[1:]
    df = f_pos[1] - f_pos[0]
    snr_squared = 4.0 * np.sum((np.abs(h_plus) ** 2 + np.abs(h_cross) ** 2) / cubic_spline_psd(f_pos) * df)
    return float(np.sqrt(snr_squared))


def _worker_compute_snr(args):
    return compute_snr(*args)


def average_snr(
    m1,
    m2,
    a,
    Tpl,
    ef,
    z,
    Tobs=None,
    psd="LISA",
    num_freq=5000,
    num_samples=16,
    seed=None,
    Phi_r0=None,
    pool=None,
):
    rng = np.random.default_rng(seed)

    sample_args = []
    for _ in range(num_samples):
        qS = np.arccos(rng.uniform(-1.0, 1.0))
        phiS = rng.uniform(0.0, 2.0 * np.pi)
        qK = np.arccos(rng.uniform(-1.0, 1.0))
        phiK = rng.uniform(0.0, 2.0 * np.pi)
        Phi_phi0 = rng.uniform(0.0, 2.0 * np.pi)
        Phi_theta0 = rng.uniform(0.0, 2.0 * np.pi)
        Phi_r0 = DEFAULT_ANGLES["Phi_r0"] if Phi_r0 is None else Phi_r0
        if num_samples == 1:
            # set to default angles for reproducibility when only one sample is requested
            qS = DEFAULT_ANGLES["qS"]
            phiS = DEFAULT_ANGLES["phiS"]
            qK = DEFAULT_ANGLES["qK"]
            phiK = DEFAULT_ANGLES["phiK"]
            Phi_phi0 = DEFAULT_ANGLES["Phi_phi0"]
            Phi_theta0 = DEFAULT_ANGLES["Phi_theta0"]
            Phi_r0 = DEFAULT_ANGLES["Phi_r0"]
        if Tobs is None:
            Tobs = Tpl
        sample_args.append(
            (
                m1,
                m2,
                a,
                Tpl,
                ef,
                z,
                Tobs,
                qS,
                phiS,
                qK,
                phiK,
                Phi_phi0,
                Phi_theta0,
                Phi_r0,
                psd,
                num_freq,
            )
        )

    if pool is None:
        results = list(map(_worker_compute_snr, sample_args))
    else:
        results = pool.map(_worker_compute_snr, sample_args)

    angles = np.asarray(np.array(sample_args)[:, 6:12],dtype=float)  # Extract angles from sample_args
    return results, angles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute EMRI SNR and averaged SNR using multiprocess.")
    parser.add_argument(
        "--psds",
        nargs="+",
        # default=["LISA", "AMADEUS", "DO-IT"],
        default=["LISA"],
        help="List of PSDs to evaluate.",
    )
    parser.add_argument("--num-freq", type=int, default=5000, help="Number of positive frequency samples.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of angle realizations for averaging.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of multiprocess workers.")
    parser.add_argument("--seed", type=int, default=2601, help="Random seed for angle sampling.")
    args = parser.parse_args()

    with h5py.File("population_data_M1_checkSNR.h5", "r+") as input_catalog:
        list_sources = list(input_catalog.keys())
        list_params = ['a1', 'dist_Gpc', 'inc', 'index', 'm1', 'm2', 'tmerger_yr', 'z']
        
        for psd in args.psds:
            start = time.time()
            for source in list_sources:
                m1 = float(input_catalog[source]["m1"][()])
                m2 = float(input_catalog[source]["m2"][()])
                a = float(input_catalog[source]["a1"][()])
                # random Tobs between 0 and 2 years
                Tobs = np.random.uniform(0.0, 2.0)
                # Tobs = float(input_catalog[source]["tmerger_yr"][()])
                ef = 0.0
                z = float(input_catalog[source]["z"][()])
                print(f"Computing SNR for source {source} with m1={m1}, m2={m2}, a={a}, Tobs={Tobs}, ef={ef}, z={z} using PSD={psd}...")
                if args.num_workers > 0:
                    with multiprocess.Pool(processes=args.num_workers) as pool:
                        snrs, angles = average_snr(
                            m1,
                            m2,
                            a,
                            Tobs,
                            ef,
                            z,
                            psd=psd,
                            num_freq=args.num_freq,
                            num_samples=args.num_samples,
                            seed=args.seed,
                            pool=pool,
                        )
                else:
                    snrs, angles = average_snr(
                        m1,
                        m2,
                        a,
                        Tobs,
                        ef,
                        z,
                        psd=psd,
                        num_freq=args.num_freq,
                        num_samples=args.num_samples,
                        seed=args.seed,
                        pool=None,
                    )
                snr_key = f"snrs_{psd}"
                angle_key = f"angles_{psd}"

                if snr_key in input_catalog[source]:
                    del input_catalog[source][snr_key]
                input_catalog[source].create_dataset(snr_key, data=snrs)

                if angle_key in input_catalog[source]:
                    del input_catalog[source][angle_key]
                input_catalog[source].create_dataset(angle_key, data=angles)
                input_catalog[source]['Tobs_realized'] = Tobs
                print(f"Source {source}: average SNR over {args.num_samples} angles = {np.mean(snrs):.3f} for PSD={psd}")
            end = time.time()
            
            print(f"PSD={psd}: computed average SNR in {end - start:.2f} seconds.")


