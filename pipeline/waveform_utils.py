import numpy as np
import os
from fastlisaresponse import ResponseWrapper
from few.waveform import GenerateEMRIWaveform
from scipy.signal import get_window

try:
    import cupy as xp
except:
    import numpy as xp

def initialize_waveform_generator(T, args, inspiral_kwargs_forward):
    backend = 'gpu' if args.use_gpu else 'cpu'
    base_wave = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", inspiral_kwargs=inspiral_kwargs_forward, force_backend=backend, sum_kwargs=dict(pad_output=True))
    orbits = "esa-trailing-orbits.h5" if args.esaorbits else "equalarmlength-orbits.h5"
    orbit_file = os.path.join(os.path.dirname(__file__), '..', 'lisa-on-gpu', 'orbit_files', orbits)
    orbit_kwargs = dict(orbit_file=orbit_file)
    tdi_kwargs_esa = dict(orbit_kwargs=orbit_kwargs, order=25, tdi="2nd generation", tdi_chan="AET")
    model = ResponseWrapper(
            base_wave, T, args.dt, 8, 7, t0=100000., flip_hx=True, use_gpu=args.use_gpu,
            remove_sky_coords=False, is_ecliptic_latitude=False, remove_garbage="zero", **tdi_kwargs_esa
        )
    return model

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

