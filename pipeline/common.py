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

