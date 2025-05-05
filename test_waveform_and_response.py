# python -m unittest test_waveform_and_response.py 
import unittest
import numpy as np
import warnings
import os

path_to_file = os.path.dirname(__file__)

from lisatools.detector import EqualArmlengthOrbits
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.utils import get_overlap

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    pass

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

YRSID_SI = 31558149.763545603


class GBWave:
    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return cp if self.use_gpu else np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):

        # get the t array
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot**2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t**2 + 1.0 / 6.0 * fddot * t**3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc


class ResponseTest(unittest.TestCase):

    def run_test(self, tdi_gen, use_gpu):
        gb = GBWave(use_gpu=use_gpu)

        T = 2.0  # years
        t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

        sampling_frequency = 0.1
        dt = 1 / sampling_frequency

        # order of the langrangian interpolation
        order = 25

        # orbit_file_esa = path_to_file + "/../../orbit_files/esa-trailing-orbits.h5"

        # orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

        index_lambda = 6
        index_beta = 7

        # orbits = EqualArmlengthOrbits(use_gpu=use_gpu)
        # orbits.configure(linear_interp_setup=True)
        # tdi_kwargs_esa = dict(
        #     orbits=orbits,
        #     order=order,
        #     tdi=tdi_gen,
        #     tdi_chan="AET",
        # )

        # gb_lisa_esa = ResponseWrapper(
        #     gb,
        #     T,
        #     dt,
        #     index_lambda,
        #     index_beta,
        #     t0=t0,
        #     flip_hx=False,  # set to True if waveform is h+ - ihx
        #     use_gpu=use_gpu,
        #     remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
        #     is_ecliptic_latitude=True,  # False if using polar angle (theta)
        #     remove_garbage=True,  # removes the beginning of the signal that has bad information
        #     **tdi_kwargs_esa,
        # )

        orbits = "equalarmlength-orbits.h5"
        orbit_file = os.path.join(os.path.dirname(__file__), 'lisa-on-gpu', 'orbit_files', orbits)
        orbit_kwargs = dict(orbit_file=orbit_file)
        # orbit_kwargs=orbit_kwargs, 
        tdi_kwargs = dict(order=order, tdi=tdi_gen, tdi_chan="AET")

        gb_lisa_esa = ResponseWrapper(
            gb,
            T,
            dt,
            index_lambda,
            index_beta,
            t0=t0,
            flip_hx=False,  # set to True if waveform is h+ - ihx
            use_gpu=use_gpu,
            remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
            is_ecliptic_latitude=True,  # False if using polar angle (theta)
            remove_garbage=True,  # removes the beginning of the signal that has bad information
            orbits = EqualArmlengthOrbits(use_gpu=use_gpu),
            **tdi_kwargs,
        )

        # define GB parameters
        A = 1.084702251e-22
        f = 2.35962078e-3
        fdot = 1.47197271e-17
        iota = 1.11820901
        phi0 = 4.91128699
        psi = 2.3290324

        beta = 0.9805742971871619
        lam = 5.22979888

        chans = gb_lisa_esa(A, f, fdot, iota, phi0, psi, lam, beta)

        return chans

    def test_tdi_1st_generation(self):

        waveform_cpu = self.run_test("1st generation", False)
        self.assertTrue(np.all(np.isnan(waveform_cpu) == False))

        if gpu_available:
            waveform_gpu = self.run_test("1st generation", True)
            mm = len(waveform_cpu) - get_overlap(
                cp.asarray(waveform_cpu),
                cp.asarray(waveform_gpu),
                use_gpu=gpu_available,
            )
            self.assertLess(np.abs(mm), 1e-10)

    def test_tdi_2nd_generation(self):

        waveform_cpu = self.run_test("2nd generation", False)
        self.assertTrue(np.all(np.isnan(waveform_cpu) == False))

        if gpu_available:
            waveform_gpu = self.run_test("2nd generation", True)
            mm = len(waveform_cpu) - get_overlap(
                cp.asarray(waveform_cpu), waveform_gpu, use_gpu=gpu_available
            )
            self.assertLess(np.abs(mm), 1e-10)

# from few.waveform import FastSchwarzschildEccentricFlux
from few.waveform import GenerateEMRIWaveform

from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux
from few.trajectory.ode import KerrEccEqFlux

from few.utils.globals import get_logger, get_first_backend

few_logger = get_logger()

best_backend = get_first_backend(FastKerrEccentricEquatorialFlux.supported_backends())
few_logger.warning("Kerr Test is running with backend {}".format(best_backend.name))

# keyword arguments for inspiral generator (Kerr Waveform)
inspiral_kwargs_Kerr = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "buffer_length": int(1e3),  # all of the trajectories will be well under len = 1000
    "func": KerrEccEqFlux,
}


class KerrWaveformTest(unittest.TestCase):
    def test_Kerr_vs_Schwarzchild(self):
        # Test whether the Kerr and Schwarzschild waveforms agree.

        wave_generator_Kerr = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux", force_backend=best_backend
        )
        wave_generator_Schwarz = GenerateEMRIWaveform(
            "FastSchwarzschildEccentricFlux", force_backend=best_backend
        )

        # parameters
        M = 1e6
        mu = 1e1
        p0 = 10.0
        e0 = 0.4

        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8

        Phi_phi0 = 1.0
        Phi_theta0 = 2.0
        Phi_r0 = 3.0

        dist = 1.0
        dt = 10.0
        T = 0.1

        Kerr_wave = wave_generator_Kerr(
            M,
            mu,
            0.0,
            p0,
            e0,
            1.0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=T,
            dt=dt,
        )
        Schwarz_wave = wave_generator_Schwarz(
            M,
            mu,
            0.0,
            p0,
            e0,
            1.0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=T,
            dt=dt,
        )

        mm = get_mismatch(Kerr_wave, Schwarz_wave, use_gpu=best_backend.uses_gpu)

        self.assertLess(mm, 1e-4)

    def test_retrograde_orbits(self):
        r"""
        Here we test that retrograde orbits and prograde orbits for a = \pm 0.7
        have large mismatches.
        """
        few_logger.info("Testing retrograde orbits")

        wave_generator_Kerr = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux", force_backend=best_backend
        )

        # parameters
        M = 1e6
        mu = 1e1
        a = 0.7
        p0 = 11.0
        e0 = 0.4

        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8

        Phi_phi0 = 1.0
        Phi_theta0 = 2.0
        Phi_r0 = 3.0

        dist = 1.0
        dt = 10.0
        T = 0.1

        Kerr_wave_retrograde = wave_generator_Kerr(
            M,
            mu,
            abs(a),
            p0,
            e0,
            -1.0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=T,
            dt=dt,
        )
        Kerr_wave_prograde = wave_generator_Kerr(
            M,
            mu,
            abs(a),
            p0,
            e0,
            1.0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=T,
            dt=dt,
        )

        mm = get_mismatch(
            Kerr_wave_retrograde, Kerr_wave_prograde, use_gpu=best_backend.uses_gpu
        )
        self.assertGreater(mm, 1e-3)

