from stableemrifisher.fisher import StableEMRIFisher
from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

try:
    import cupy as xp
except:
    xp = np


import numpy as np
from scipy import integrate
from scipy.interpolate import splrep, splev, RegularGridInterpolator, interp1d
from scipy.constants import c

c /= 1000.  # 2.99792458e+05 # in km/s


def h(z, Omega_m=0.3065, w0=-1., wa=0.):
    """
    Returns dimensionless redshift-dependent hubble parameter.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction
    Dynamical dark energy: w(a) = w0+wa(1-a)

    Returns
    -------
    dimensionless h(z) = sqrt(Omega_m*(1+z)^3 + Omega_Lambda
    *(1+z)^[3(1+w0+wa)]*e^[-3*wa*z/(1+z)])
    """
    Omega_Lambda = (1-Omega_m)
    return np.sqrt(Omega_m*(1+z)**3 + Omega_Lambda* (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z)))


def dcH0overc(z, Omega_m=0.3065, w0=-1., wa=0.):
    """
    Returns dimensionless combination dc*H0/c
    given redshift and matter fraction.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction
    Dynamical dark energy: w(a) = w0+wa(1-a)

    Returns
    -------
    dimensionless combination dc*H0/c = \int_0^z dz'/h(z')
    """
    integrand = lambda zz: 1./h(zz, Omega_m, w0, wa)

    if np.size(z)>1:
        if np.size(np.where(z<=0))>0:
            raise ValueError('Negative redshift input!')
        result = np.array([integrate.quad(integrand, 0, zi)[0] for zi in z])
    else:
        if z<=0:
            raise ValueError('Negative redshift input!')
        result = integrate.quad(integrand, 0, z)[0]  # in km/s

    return result


def dLH0overc(z, Omega_m=0.3065, w0=-1., wa=0.):
    """
    Returns dimensionless combination dL*H0/c
    given redshift and matter fraction.

    Parameters
    ----------
    z : redshift
    Omega_m : matter fraction

    Returns
    -------
    dimensionless combination dL*H0/c = (1+z) * \int_0^z dz'/h(z')
    """
    return (1+z)*dcH0overc(z, Omega_m, w0, wa)

class standard_cosmology(object):
    """
    from gwcosmo
    """
    def __init__(self, H0=70, Omega_m=0.3065, w0=-1., wa=0., zmax=10.0, zmin=1.e-5, zbin=5000):

        self.c = c
        self.H0 = H0
        self.Omega_m = Omega_m
        self.w0 = w0
        self.wa = wa
        self.zmax = zmax
        self.zmin = zmin
        self.zbin = zbin
        self.z_array = np.logspace(np.log10(self.zmin), np.log10(self.zmax), self.zbin)

        # Interpolation of z(dL)
        self.dlH0overc_z_arr = np.array([dLH0overc(z, Omega_m=self.Omega_m, w0=self.w0, wa=self.wa)
                        for z in self.z_array])
        self.dlH0overc_of_z = splrep(self.z_array, self.dlH0overc_z_arr)

    def update_parameters(self,param_dict):
        """
        Update values of cosmological parameters.
        Key in param_dict: H0
        """
        if 'H0' in param_dict:
            if param_dict['H0'] != self.H0:
                self.H0 = param_dict['H0']

    def dl_zH0(self, z):
        """
        Returns luminosity distance given redshift

        Parameters
        ----------
        z : redshift

        Returns
        -------
        luminosity distance, dl (in Mpc)
        """
        return splev(z, self.dlH0overc_of_z, ext=3)*c/self.H0


def get_covariance_matrix(
        model, 
        parameters, 
        dt=10., 
        T=1., 
        psd=None, 
        use_gpu=False, 
        inspiral_kwargs=None, 
        sum_kwargs=None, 
        outdir=None, 
        params=None,
        deltas=None,
        log_e=False,
        **kwargs,
    ):

    if inspiral_kwargs is None:
        inspiral_kwargs = {}

    if sum_kwargs is None:
        sum_kwargs = {}

    sum_kwargs["pad_output"] = True

    #varied parameters
    if params is None:
        param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])

    popinds = []
    try:
        if model.waveform_gen.waveform_generator.descriptor == "eccentric":
            popinds.append(5)
            popinds.append(12)
        if model.waveform_gen.waveform_generator.background == "Schwarzschild":
            popinds.append(2)
    except:
        if model.waveform_generator.descriptor == "eccentric":
            popinds.append(5)
            popinds.append(12)
        if model.waveform_generator.background == "Schwarzschild":
            popinds.append(2)

    param_names = np.delete(param_names, popinds).tolist()

    if outdir is not None:
        Path(outdir).mkdir(exist_ok=True, parents=True)

    #initialization
    fish = StableEMRIFisher(*parameters, dt=dt, T=T, EMRI_waveform_gen=model, noise_model=psd, noise_kwargs=dict(TDI="TDI2"),
                param_names=param_names, stats_for_nerds=False, use_gpu=use_gpu, deltas=deltas, der_order=4., Ndelta=20, log_e=log_e, 
                filename=outdir, **kwargs)

    #execution
    SNR = fish.SNRcalc_SEF()
    fim = fish()

    if log_e:
        jac = np.diag([1, 1, 1, 1, 1/parameters[4], 1, 1, 1, 1, 1, 1, 1]) #if working in log_e space apply jacobian to the fisher matrix
        fim = jac.T @ fim @ jac
        
    cov = np.linalg.inv(fim)
    return cov, fish, SNR

def draw_sources(fix_params, N_per_source, seed=0):
    np.random.seed(seed)
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

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="Waveform model to use")
    parser.add_argument("psd_file", help="Path to a file containing PSD frequency-value pairs")
    parser.add_argument("M", help="MBH mass in solar masses", type=float)
    parser.add_argument("mu", help="CO mass in solar masses", type=float)
    parser.add_argument("a", help="MBH dimensionless spin parameter", type=float)
    parser.add_argument("p0", help="Initial dimensionless semi-latus rectum of inspiral", type=float)
    parser.add_argument("e0", help="Initial eccentricity of inspiral", type=float)
    parser.add_argument("xI0", help="Initial cos(inclination) of inspiral", type=float)
    parser.add_argument("dist", help="Luminosity distance to source.", type=float)
    parser.add_argument("qS", help="Source polar sky angle", type=float)
    parser.add_argument("phiS", help="Source ecliptic longitude", type=float)
    parser.add_argument("qK", help="Source polar spin angle", type=float)
    parser.add_argument("phiK", help="Source azimuthal spin angle", type=float)
    parser.add_argument("Phi_phi0", help="Initial azimuthal phase of inspiral", type=float)
    parser.add_argument("Phi_theta0", help="Initial polar phase of inspiral", type=float)
    parser.add_argument("Phi_r0", help="Initial radial phase of inspiral", type=float)
    parser.add_argument("dt", help="Sampling cadence in seconds", type=float)
    parser.add_argument("T", help="Waveform duration in years.", type=float)

    parser.add_argument("--use_gpu", help="Whether to use GPU for FIM computation", action="store_true")
    parser.add_argument("--deltas_file", help="Path to a file containing the stable delta values for this source. If not given, solves for stable deltas", action="store_const")

    args = parser.parse_args()

    # deltas = np.loadtxt(args.deltas_file)

    parameters = [
        args.M,
        args.mu,
        args.a,
        args.p0,
        args.e0,
        args.xI0,
        args.dist,
        args.qS,
        args.phiS,
        args.qK,
        args.phiK,
        args.Phi_phi0,
        args.Phi_theta0,
        args.Phi_r0,
    ]

    base_wave = GenerateEMRIWaveform(args.model, use_gpu=args.use_gpu)

    # order of the langrangian interpolation
    order = 10

    orbit_file_esa = "./orbit_files/esa-trailing-orbits.h5"

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

    psdf, psdv = np.load(args.psd_file).T
    psd_interp = CubicSpline(psdf, psdv)
    psd_wrap = lambda f, **kwargs: psd_interp(f)
    
    cov, fisher_object = get_covariance_matrix(model, parameters, dt=args.dt, T=args.T, psd=psd_wrap, use_gpu=args.use_gpu)

    print(cov)
