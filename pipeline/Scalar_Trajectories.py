import os
import sys

import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt


# Import relevant EMRI packages
import few
from few.waveform import FastSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.trajectory.ode import PN5, SchwarzEccFlux, KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
#from few.utils.utility import get_separatrix, get_p_at_t (old FEW version)

from few.summation.directmodesum import DirectModeSum
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.fdinterp import FDInterpolatedModeSum
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.modeselector import ModeSelector, NeuralModeSelector

from few.trajectory.ode.base import ODEBase
from few.trajectory.ode import SchwarzEccFlux
#from few.utils.utility import get_fundamental_frequencies, get_separatrix, ELQ_to_pex (old FEW version)
from few.utils.geodesic import get_fundamental_frequencies, get_separatrix, ELQ_to_pex
from few.utils.mappings.jacobian import ELdot_to_PEdot_Jacobian

from few.waveform.base import SphericalHarmonicWaveformBase
from few.amplitude.romannet import RomanAmplitude

from few.utils.baseclasses import (
    SchwarzschildEccentric,
    KerrEccentricEquatorial,
    Pn5AAK,
    ParallelModuleBase,
)

from typing import Union, Optional
from multispline.spline import BicubicSpline, TricubicSpline, CubicSpline
from pathlib import Path

## ====================================================================# 
## ================== KERR CIRCULAR ===================================#
## ====================================================================#  
class KerrCircEqFluxScalar(KerrEccEqFlux):
    @staticmethod
    def load_scalar_flux_table(filename):
        data = np.loadtxt(filename)
        a_col = data[:, 0]
        p_col = data[:, 1]
        F_hor = data[:, 4]
        F_inf = data[:, 5]
        z_col = data[:, 7]
        u_col = data[:, 8]

        F_total_col = (F_hor + F_inf)

        Omega_phi = 1.0 / (p_col**1.5 + a_col)

        Epn = 1.0 / (12. * p_col**(4.))

        F_total_with_PN_rescaling = (F_total_col - Epn) * p_col**(6.)
    
        return z_col, u_col, F_total_with_PN_rescaling

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        BASE_DIR = Path(__file__).resolve().parent
        data_file_A = BASE_DIR / "tabA_ScalarKerr.dat"
        data_file_B = BASE_DIR / "tabB_ScalarKerr.dat"
        z_A, u_A, F_A = self.load_scalar_flux_table(data_file_A) 
        z_B, u_B, F_B = self.load_scalar_flux_table(data_file_B)

        z_A_unique_vals = np.unique(z_A)    
        u_A_unique_vals = np.unique(u_A)   
        z_B_unique_vals = np.unique(z_B)    
        u_B_unique_vals = np.unique(u_B)   

        F_A_grid = F_A.reshape((len(z_A_unique_vals), len(u_A_unique_vals)))
        F_B_grid = F_B.reshape((len(z_B_unique_vals), len(u_B_unique_vals)))

        self.Fphi_interp_A = BicubicSpline(z_A_unique_vals, u_A_unique_vals, F_A_grid)
        self.Fphi_interp_B = BicubicSpline(z_B_unique_vals, u_B_unique_vals, F_B_grid)   

    def compute_Edot_phi(self, p):
        a = self.a
        p_sep = get_separatrix(a, 0.,1.0)
        amin = -999/1000
        amax = 999/1000

        chi_min = (1 - amax)**(1/3)
        chi_max = (1 - amin)**(1/3)

        delta_pAmin = 1e-3
        delta_pAmax = 9.0 + delta_pAmin
        Cp = delta_pAmax - 2 * delta_pAmin
        CDelta = np.log(delta_pAmax - delta_pAmin)
        pminA = delta_pAmin + p_sep
        pmaxA = delta_pAmax + p_sep

        delta_pBmin = delta_pAmax
        delta_pBmax = 200.0 
        pminB = delta_pBmin + p_sep
        pmaxB = delta_pBmax + p_sep

        z = ((1 - a)**(1/3) - chi_min) / (chi_max - chi_min)

        if p < pmaxA:
            u = (np.log(p - p_sep + Cp) - CDelta) / np.log(2)
            F = self.Fphi_interp_A(z, u)
        else:
            u = (delta_pBmin**(-0.5) - (p - p_sep)**(-0.5)) / (delta_pBmin**(-0.5) - (pmaxB - p_sep)**(-0.5))
            F = self.Fphi_interp_B(z, u)
        
        Omega_phi = 1.0 / (p**1.5 + a)
        Epn = 1.0 / (12. * p**(4.))

        Edot_phi = (F * p**(- 6.) + Epn)

        return Edot_phi

    def modify_rhs(self, ydot, y):
        # Map to flux grid coordinates
        Edot_phi = self.compute_Edot_phi(y[0])

        # Scale by scalar charge squared
        q_s2 = self.additional_args[2] if hasattr(self, "additional_args") and len(self.additional_args) > 0 else 0.0

        Edot_phi *= q_s2
        
        sqrt_r = np.sqrt(y[0])
        r_32 = y[0]**1.5
        r2 = y[0]**2
        numerator = -3 * self.a**2 + 8 * self.a * sqrt_r + (-6 + y[0]) * y[0]
        term1 = 2 * self.a * y[0] + (-3 + y[0]) * r_32
        term2 = 2 * self.a * r_32 + (-3 + y[0]) * r2
        denominator = 2 * term1 * np.sqrt(term2)
        dE_dp = numerator/denominator

        ydot[0] = (ydot[0] - Edot_phi / dE_dp)


## ==========================================================================================#
## ================== MASSIVE ===============================================================#
## ==========================================================================================#       
class KerrCircEqFluxMassiveScalar18(KerrEccEqFlux):
    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)
        
        try:
            data = np.loadtxt("/home/people/sbarsanti/few/data_few/pdot_mu0018_lmax1.dat") #divided by 4 already
        except KeyError:
            raise ValueError("Unsupported mu value. Values supported: 0.018 or 0.036")
      
        p_vals = np.unique(data[:, 0])
        a_vals = np.unique(data[:, 1])
        mu_vals= np.unique(data[:, 2])

        Np, Na, Nmu = len(p_vals), len(a_vals), len(mu_vals)

        pdot_grid = np.empty((Np, Na, Nmu))

        for row in data:
            i = np.where(p_vals  == row[0])[0][0]
            j = np.where(a_vals  == row[1])[0][0]
            k = np.where(mu_vals == row[2])[0][0]
            pdot_grid[i, j, k] = row[3]

        #lo sto definendo qui Edot_interp. E poi lo posso richiamare! 
        self.pdot_interp_scalar =  TricubicSpline(p_vals, a_vals, mu_vals, pdot_grid)      
    
    def interpolate_flux_scalar_massive(
              self, p: float
              ) -> float:

        a = self.a 
        mu = self.additional_args[3]
        pdot_scal = - self.additional_args[2]*self.pdot_interp_scalar(p,a,mu)
         
        return pdot_scal
    
    def modify_rhs(self, ydot, y):
        pdotscal = self.interpolate_flux_scalar_massive(y[0])
        ydot[0] = ydot[0] + pdotscal


class KerrCircEqFluxMassiveScalar36(KerrEccEqFlux):
    def __init__(self, *args, use_ELQ: bool = False, **kwargs):
        super().__init__(*args, use_ELQ=use_ELQ, **kwargs)
        
        try:
            data = np.loadtxt("/home/people/sbarsanti/few/data_few/pdot_mu0036_lmax1.dat")
        except KeyError:
            raise ValueError("Unsupported mu value. Values supported: 0.018 or 0.036")
      
        p_vals = np.unique(data[:, 0])
        a_vals = np.unique(data[:, 1])
        mu_vals= np.unique(data[:, 2])

        Np, Na, Nmu = len(p_vals), len(a_vals), len(mu_vals)

        pdot_grid = np.empty((Np, Na, Nmu))

        for row in data:
            i = np.where(p_vals  == row[0])[0][0]
            j = np.where(a_vals  == row[1])[0][0]
            k = np.where(mu_vals == row[2])[0][0]
            pdot_grid[i, j, k] = row[3]

        #lo sto definendo qui Edot_interp. E poi lo posso richiamare! 
        self.pdot_interp_scalar =  TricubicSpline(p_vals, a_vals, mu_vals, pdot_grid)      
    
    def interpolate_flux_scalar_massive(
              self, p: float
              ) -> float:

        a = self.a 
        mu = self.additional_args[3]
        pdot_scal = - self.additional_args[2]*self.pdot_interp_scalar(p,a,mu)
         
        return pdot_scal
    
    def modify_rhs(self, ydot, y):
        pdotscal = self.interpolate_flux_scalar_massive(y[0])
        ydot[0] = ydot[0] + pdotscal