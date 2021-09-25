'''
 Physics Constants
'''
import numpy as np

epsilon0 = 8.8541878176e-12       # vacuum permittivity
mu0      = 4* np.pi*1e-7          # vacuum permiability
c        = 2.99792458e8           # speed of light (m/s)
c_cgs    = c*100.                 # speed of light (cm/s)
q0       = 1.60217662e-19         # electron charge
q0_cgs   = 4.80320427e-10         # electron charge(cgs)
k_B      = 1.380649e-23     # Boltzmann Constant (J/K)
alpha_fine  = 0.0072973525693     # fine structure constant
Da       = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

mass_electron = Da/1822.8884845
mass_hydrogen = Da*1.008 
mass_proton = 1.67262192369e-27
