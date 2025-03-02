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
    
    def dl_dz(self, z):
        """
        Returns luminosity derivative wrt redshift given redshift

        Parameters
        ----------
        z : redshift

        Returns
        -------
        redshift
        """
        return splev(z, self.dlH0overc_of_z, ext=3, der=1)*c/self.H0
    
    def transform_mass_uncertainty(self, m, sigma_m, z, sigma_l):
        """
        Transform mass uncertainty from detector frame mass to source frame mass

        Parameters
        ----------
        m : detector frame mass
        sigma_m : mass uncertainty
        z : redshift
        sigma_l [Mpc] : luminosity distance uncertainty
        """
        sigma_dz = sigma_l / np.abs(cosmo.dl_dz(z))
        sigma_Msource = np.sqrt( (sigma_m / (1+z))**2 + (m/(1+z**2) * sigma_dz)**2 )
        return sigma_Msource
    
    def jacobian(self, M_s, mu_s, z):
        """Jacobian to obtain source frame Fisher matrix from detector frame Fisher matrix. GammaNew = J^T Gamma J

        Args:
            M_s (float): Source frame central mass of the binary in solar masses
            mu_s (float): secondary mass of the binary in solar masses
            dz_dl (float): Derivative of redshift with respect to luminosity distance
            z (float): Redshift of the source

        Returns:
            np.array: Jacobian matrix
        """
        dz_dl = 1. / self.dl_dz(z)
        first_row =  np.array([(1+z), 0.0,   0.0, 0.0, 0.0, M_s * dz_dl,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        second_row = np.array([0.0,   (1+z), 0.0, 0.0, 0.0, mu_s * dz_dl, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        third_row =  np.array([0.0,   0.0,   1.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fourth_row = np.array([0.0,   0.0,   0.0, 1.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fifth_row =  np.array([0.0,   0.0,   0.0, 0.0, 1.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sixth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 1.0,          0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        seventh_row = np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        eighth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        ninth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        tenth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        eleventh_row = np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        twelfth_row = np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        J = np.array([first_row, second_row, third_row, fourth_row, fifth_row, sixth_row, seventh_row, eighth_row, ninth_row, tenth_row, eleventh_row, twelfth_row])
        # print("shape of J: ", J.shape)
        return J

if __name__ == "__main__":

    # test transformation of mass uncertainty
    cosmo = standard_cosmology(zmax=10.0, zmin=1.e-3, zbin=100000)
    z = 0.1
    l = cosmo.dl_zH0(z)
    print("Luminosity distance [Mpc]: ", l, "Redshift", z)
    m = 1.e6
    msource = m / (1+z)
    sigma_m_values = np.logspace(-5, -2, 4) * m
    sigma_l_values = np.logspace(-3, -0.5, 20) * l
    sigma_msource_values = np.zeros((len(sigma_m_values), len(sigma_l_values)))

    for i, sigma_m in enumerate(sigma_m_values):
        for j, sigma_l in enumerate(sigma_l_values):
            sigma_msource_values[i, j] = cosmo.transform_mass_uncertainty(m, sigma_m, z, sigma_l)

    plt.figure(figsize=(12, 6))

    # First subplot: Relative uncertainty in source mass
    plt.subplot(1, 2, 1)
    for i, sigma_m in enumerate(sigma_m_values):
        plt.plot(sigma_l_values / l, sigma_msource_values[i, :] / msource, label=f'Sigma_m/m={sigma_m/m:.1e}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative uncertainty in luminosity distance')
    plt.ylabel('Relative uncertainty in source mass')
    plt.legend()
    plt.grid(True)
    plt.title('Relative uncertainty in source mass')

    # Second subplot: Ratio of relative precision
    plt.subplot(1, 2, 2)
    for i, sigma_m in enumerate(sigma_m_values):
        ratio = (sigma_msource_values[i, :] / msource) / (sigma_m / m)
        plt.plot(sigma_l_values / l, ratio, label=f'Sigma_m/m={sigma_m/m:.1e}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative uncertainty in luminosity distance')
    plt.ylabel('Ratio of relative precision (source/detector)')
    plt.legend()
    plt.grid(True)
    plt.title('Ratio of relative precision')

    plt.tight_layout()
    plt.savefig('uncertainty_msource.png')
    # print("Sigma Msource: ", sigma_msource, "Sigma Mdetector: ", sigma_m)
    # # relative uncertainty
    # print("Relative uncertainty in source: ", sigma_msource / msource, " and detector mass ",sigma_m / m)
    # print("Sigma L: ", sigma_l, "Sigma dz: ", sigma_l / np.abs(cosmo.dl_dz(z)))

    # plot dL(z)
    cosmo = standard_cosmology(zmax=100.0, zmin=1.e-9, zbin=100000)
    z = np.logspace(np.log10(1e-9), -2, 1000)
    dl = np.array([cosmo.dl_zH0(zi) for zi in z])/1e3
    plt.figure()
    plt.plot(z, dl)
    # plot horizontal line at 8 kpc
    plt.axhline(8e-9, color='r', label='8 kpc')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('z')
    plt.ylabel('dL (Gpc)')

    galaxies = {
    # Local Group
    "Andromeda (M31)": (0.00044, 0.78, 1.1e8),  # Bender et al. 2005
    "Triangulum (M33)": (0.00059, 0.86, None),  # No confirmed SMBH (Gebhardt et al. 2001)
    # "Milky Way": (0.00000, 0.00, 4.3e6),  # Sgr A*, Gravity Collab. 2019
    
    # Nearby galaxies
    "Centaurus A": (0.00183, 3.8, 5.5e7),  # Silge et al. 2005
    "Messier 81 (M81)": (0.00086, 3.6, 7.0e7),  # Devereux et al. 2003
    # "Messier 87 (M87)": (0.0043, 16.4, 6.5e9),  # EHT Collaboration 2019
    "Sculptor Galaxy (NGC 253)": (0.0008, 3.5, None),
    "Whirlpool Galaxy (M51)": (0.0015, 8.6, 1e6),
    
    # Intermediate redshift galaxies
    # "Sombrero Galaxy (M104)": (0.0034, 9.6, 6.4e8),  # Kormendy et al. 1996
    # "3C 273 (Quasar)": (0.158, 750, 8.9e8),  # Peterson et al. 2004
    # "Cloverleaf Quasar": (2.56, None, 1.0e9),  # Lensed system
    
    # High-redshift galaxies
    # "GN-z11": (10.957, 32000, None),  # Most distant confirmed galaxy (Oesch et al. 2016)
    # "J0313-1806": (7.64, None, 1.6e9),  # Earliest quasar (Wang et al. 2021)
    # "J1342+0928": (7.54, None, 8.0e8),  # Quasar (Bañados et al. 2018)
    # "J1120+0641": (7.08, None, 2.0e9),  # Mortlock et al. 2011
    }


    # Extract data for plotting
    # galaxy_redshifts = np.array([galaxies[name][0] for name in galaxies])
    # galaxy_distances = np.array([cosmo.dl_zH0(galaxies[name][0]) for name in galaxies])/1e3

    # Plot
    # Define markers and colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_']
    colors = plt.cm.viridis(np.linspace(0, 1, len(galaxies)))

    # Annotate galaxies with different colors and markers
    for i, (name, (z, d, mass)) in enumerate(galaxies.items()):
        labmass = f", M={int(galaxies[name][2]/1e6)}" + r"$\times 10^6 M_\odot$" if galaxies[name][2] is not None else ""
        plt.scatter(galaxies[name][0], cosmo.dl_zH0(galaxies[name][0])/1e3, label=name + labmass, 
                    color=colors[i], marker=markers[i % len(markers)], zorder=3)
        # plt.annotate(name, (z, cosmo.dl_zH0(z)/1e3), textcoords="offset points", xytext=(-10, 5), ha='right', fontsize=9)


    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('dL_z.png')
