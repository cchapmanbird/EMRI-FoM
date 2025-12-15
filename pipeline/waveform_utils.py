import numpy as np
import os
from fastlisaresponse import ResponseWrapper
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import KerrEccEqFlux
from scipy.signal import get_window
from few.utils.constants import *
try:
    import cupy as xp
except:
    import numpy as xp

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
            -3 * pow(self.a, 3)
            + pow(self.a, 2) * (8 - 3 * y[0]) * np.sqrt(y[0])
            + (-6 + y[0]) * pow(y[0], 2.5)
            + 3 * self.a * y[0] * (-2 + 3 * y[0])
        ) / (2.0 * pow(2 * self.a + (-3 + y[0]) * np.sqrt(y[0]), 1.5) * pow(y[0], 1.75))
        # transform back to pdot from Ldot and add GW contribution
        ydot[0] = ydot[0] + LdotAcc / dL_dp


class wave_windowed_truncated():
    def __init__(self, wave_gen, xp, t0=100000.0):
        self.wave_gen = wave_gen
        # total length from the waveform generator
        N = round(wave_gen.Tobs * YRSID_SI / wave_gen.dt)
        N_0 = round(t0 / wave_gen.dt)
        print("Creating windowed truncated waveform with N =", N, "and response truncation N_0 =", N_0)
        taper_duration = 86400.0 * 1 # one day
        taper_length = round(taper_duration / wave_gen.dt)
        hann = np.hanning(2*taper_length)
        
        sig_tapered = np.ones(N)
        sig_tapered[:N_0] *= 0.0
        sig_tapered[N_0:N_0 + taper_length] *= hann[:taper_length]
        sig_tapered[-taper_length-N_0:-N_0] *= hann[-taper_length:]
        sig_tapered[-N_0:] *= 0.0
        self.window = xp.asarray(sig_tapered)
        self.window = xp.atleast_2d(self.window)
        self.xp = xp
        # quick plot to check window
        # import matplotlib.pyplot as plt
        # plt.figure(); plt.plot(np.arange(N) * wave_gen.dt / 86400.0, xp.asnumpy(self.window)); plt.xlabel("Time (days)"); plt.ylabel("Window amplitude"); plt.savefig("waveform_window.png")
        print("Initialized wave_windowed_truncated with N =", N, "dt =", wave_gen.dt, "power window =", np.sum(self.window**2)/N)
    
    def __call__(self, *args, **kwargs):
        wave = self.xp.asarray(self.wave_gen(*args, **kwargs))
        # breakpoint()
        # import matplotlib.pyplot as plt
        # N_0 = round(100000.0 / self.wave_gen.dt)
        # plt.figure(); 
        # plt.plot(xp.asnumpy(wave)[0]); 
        # plt.plot(xp.asnumpy(wave)[0]*xp.asnumpy(self.window)[0]); 
        # # plt.axvline(N_0, color='r'); 
        # plt.xlim(N_0 - 86400 / self.wave_gen.dt, N_0 + 86400 / self.wave_gen.dt);  
        # plt.xlim(len(wave[0]) - (N_0 + 86400 / self.wave_gen.dt), len(wave[0]) - (N_0 - 86400 / self.wave_gen.dt));  
        # plt.xlabel("Time (days)"); plt.ylabel("Window amplitude"); plt.savefig("waveform_window.png")
        
        # apply window
        wave = wave * self.window
        return wave

    def __getattr__(self, name):
        # Forward attribute access to base_wave
        return getattr(self.wave_gen, name)

def initialize_waveform_generator(T, dt, inspiral_kwargs, esaorbits=True, use_gpu=True, t0=100000.0):
    backend = 'gpu' if use_gpu else 'cpu'
    print("inspiral_kwargs:",inspiral_kwargs)
    temp_wave = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", inspiral_kwargs=inspiral_kwargs, force_backend=backend, sum_kwargs=dict(pad_output=True))
    orbits = "esa-trailing-orbits.h5" if esaorbits else "equalarmlength-orbits.h5"
    orbit_file = os.path.join(os.path.dirname(__file__), '..', 'lisa-on-gpu', 'orbit_files', orbits)
    orbit_kwargs = dict(orbit_file=orbit_file)
    tdi_kwargs_esa = dict(orbit_kwargs=orbit_kwargs, order=25, tdi="2nd generation", tdi_chan="AE")
    model = ResponseWrapper(
            temp_wave, T, dt, 8, 7, t0=t0, flip_hx=True, use_gpu=use_gpu,
            remove_sky_coords=False, is_ecliptic_latitude=False, remove_garbage="zero", **tdi_kwargs_esa
        )
    # base_wave = wave_windowed_truncated(model, xp, t0)
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
