import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import numpy as np

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value, FlatLambdaCDM
ref_cosmo = Planck18

def get_redshift(distance):
    return (z_at_value(ref_cosmo.luminosity_distance, distance * u.Gpc )).value

def get_distance(redshift):
    return ref_cosmo.luminosity_distance(redshift).to(u.Gpc).value

class CosmoInterpolator:
    """
    Class to interpolate cosmological parameters.
    """
    def __init__(self, min_z=1e-3, max_z=15.0, num_points=10000):
        self.min_z = min_z
        self.max_z = max_z
        self.num_points = num_points

        # Create a grid of redshifts and corresponding luminosity distances
        self.redshifts = np.linspace(self.min_z, self.max_z, self.num_points)
        self.luminosity_distances = np.array([get_distance(z) for z in self.redshifts])
        self.min_luminosity_distance = np.min(self.luminosity_distances)
        self.max_luminosity_distance = np.max(self.luminosity_distances)
        self.luminosity_distance_interpolator = CubicSpline(self.redshifts, self.luminosity_distances)
        self.redshift_interpolator = CubicSpline(self.luminosity_distances, self.redshifts)
        dz_dl = self.redshift_interpolator.derivative()(self.luminosity_distances)
        self.get_dz_dl_interp = CubicSpline(self.redshifts, dz_dl)
        # plt.figure(figsize=(8, 6))
        # plt.plot(np.linspace(self.min_z, self.max_z, self.num_points*10), self.get_dz_dl_interp(np.linspace(self.min_z, self.max_z, self.num_points*10)))
        # plt.savefig('dz_dl.png')
        

    def test_relationship(self):
        """
        Test the relationship between luminosity distance and redshift.
        """
        # Generate a range of redshifts
        z_values = np.random.uniform(self.min_z, self.max_z, 1000)
        dL_true = np.asarray([get_distance(z) for z in z_values])
        z_interp = self.redshift_interpolator(dL_true)
        dL_interp = self.luminosity_distance_interpolator(z_values)
        # plot relationship
        plt.figure(figsize=(8, 6))
        plt.plot(z_values, np.abs(1-dL_interp/dL_true), '.' ,label='Interpolated dL', color='blue')
        plt.plot(z_values, np.abs(1-z_interp/z_values), '.' ,label='Interpolated z', color='red')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Relative difference')
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlim(self.min_z, self.max_z)
        # plt.ylim(self.min_luminosity_distance, self.max_luminosity_distance)
        plt.savefig('luminosity_distance_vs_redshift.png')

    def get_redshift(self, luminosity_distance):
        """
        Get the redshift corresponding to a given luminosity distance.
        """
        return self.redshift_interpolator(luminosity_distance)
    
    def get_luminosity_distance(self, redshift):
        """
        Get the luminosity distance corresponding to a given redshift.
        """
        return self.luminosity_distance_interpolator(redshift)

    def get_dlum_dz(self, redshift):
        """
        Get the derivative of luminosity distance with respect to redshift.
        """
        return self.luminosity_distance_interpolator.derivative()(redshift)
    
    def get_dz_dlum(self, luminosity_distance):
        """
        Get the derivative of redshift with respect to luminosity distance.
        """
        return self.redshift_interpolator.derivative()(luminosity_distance)

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
        dz_dl = self.get_dz_dl_interp(z)
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
    
    def jacobian_powerlaw(self, M_s, mu_s, z):
        """Jacobian to obtain source frame Fisher matrix from detector frame Fisher matrix. GammaNew = J^T Gamma J

        Args:
            M_s (float): Source frame central mass of the binary in solar masses
            mu_s (float): secondary mass of the binary in solar masses
            dz_dl (float): Derivative of redshift with respect to luminosity distance
            z (float): Redshift of the source

        Returns:
            np.array: Jacobian matrix
        """
        dz_dl = self.get_dz_dl_interp(z)
        first_row =  np.array([(1+z), 0.0,   0.0, 0.0, M_s * dz_dl,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        second_row = np.array([0.0,   (1+z), 0.0, 0.0, mu_s * dz_dl, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        third_row =  np.array([0.0,   0.0,   1.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0])
        fourth_row = np.array([0.0,   0.0,   0.0, 1.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0])
        fifth_row =  np.array([0.0,   0.0,   0.0, 0.0, 1.0, 0.0,          0.0, 0.0, 0.0, 0.0, 0.0])
        sixth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 1.0,          0.0, 0.0, 0.0, 0.0, 0.0])
        seventh_row = np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          1.0, 0.0, 0.0, 0.0, 0.0])
        eighth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 1.0, 0.0, 0.0, 0.0])
        ninth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 1.0, 0.0, 0.0])
        tenth_row =  np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 1.0, 0.0])
        eleventh_row = np.array([0.0,   0.0,   0.0, 0.0, 0.0, 0.0,          0.0, 0.0, 0.0, 0.0, 1.0])
        J = np.array([first_row, second_row, third_row, fourth_row, fifth_row, sixth_row, seventh_row, eighth_row, ninth_row, tenth_row, eleventh_row])
        # print("shape of J: ", J.shape)
        return J


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
        sigma_dz = sigma_l  * np.abs(self.get_dz_dlum(self.get_luminosity_distance(z)))
        sigma_Msource = np.sqrt( (sigma_m / (1+z))**2 + (m/(1+z)**2 * sigma_dz)**2 )
        return sigma_Msource

CosmoInt = CosmoInterpolator()

if __name__ == "__main__":

    # test transformation of mass uncertainty
    cosmo = CosmoInterpolator()
    CosmoInt.test_relationship()
    z = 0.1
    l = cosmo.get_luminosity_distance(z)
    print("Luminosity distance [Gpc]: ", l, "Redshift", z)
    # z = 1.0
    # l = cosmo.get_luminosity_distance(z)
    # print("Luminosity distance [Gpc]: ", l, "Redshift", z)
    
    m = 1.e6
    msource = m / (1+z)
    sigma_m_values = np.logspace(-5, -2, 4) * m
    sigma_l_values = np.logspace(-3, -0.5, 20) * l
    sigma_msource_values = np.zeros((len(sigma_m_values), len(sigma_l_values)))
    print("of order one = ",np.abs(cosmo.get_dz_dlum(cosmo.get_luminosity_distance(z)))/ z * l)
    for i, sigma_m in enumerate(sigma_m_values):
        for j, sigma_l in enumerate(sigma_l_values):
            Gamma = np.eye(12)
            Gamma[0,0] = Gamma[0,0] / sigma_m**2
            Gamma[5,5] = Gamma[5,5] / sigma_l**2
            J = cosmo.jacobian(msource, 10.0, z)
            Cov = np.linalg.inv(J.T @ Gamma @ J)
            new_sigma_m = np.sqrt(Cov[0, 0])
            sigma_msource_values[i, j] = new_sigma_m # 
    print("check jacobian transformation: ", np.abs(1-new_sigma_m/cosmo.transform_mass_uncertainty(m, sigma_m, z, sigma_l)))

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

    # Plot rho = (1 + z)**(5/6) / D_L(z) over a range of redshifts
    z_plot = np.logspace(-2, 1., 1000)
    dL_plot = cosmo.get_luminosity_distance(z_plot)
    rho = (1 + z_plot)**(5/6) / dL_plot

    plt.figure(figsize=(8, 6))
    plt.plot(z_plot, rho)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Redshift (z)')
    plt.ylabel(r'$\rho = (1+z)^{5/6} / D_L(z)$ [1/Gpc]')
    plt.title(r'$\rho$ vs Redshift')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rho_vs_redshift.png')
    print(np.logspace(np.log10(0.05), np.log10(1.5), 5))
