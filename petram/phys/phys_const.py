#
# Physics Constants
#
import numpy as np

epsilon0 = 8.8541878176e-12       # vacuum permittivity
mu0 = 4 * np.pi*1e-7          # vacuum permiability
c = 2.99792458e8           # speed of light (m/s)
c_cgs = c*100.                 # speed of light (cm/s)
q0 = 1.60217662e-19         # electron charge
q0_cgs = 4.80320427e-10         # electron charge(cgs)
k_B = 1.380649e-23     # Boltzmann Constant (J/K)
alpha_fine = 0.0072973525693     # fine structure constant
Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

mass_electron = Da/1822.8884845
mass_hydrogen = Da*1.008
mass_proton = 1.67262192369e-27

#
# atomic mass in Da unit
#   information here is mostly from https://pubchem.ncbi.nlm.nih.gov/ptable/atomic-mass, which
#   lists rounded numbers. For some atomes, more digits are given manually
#
#
massu = {"e": 1./1822.8884845,
         "H": 1.00797,
         "D": 2.01410177811,
         "T": 3.01604928,
         "He": 4.0026,
         "He3": 3.0160293,
         "Li": 7.016003434,
         "Li6": 6.0151228874,
         "Be": 9.012183,
         "B": 10.81,
         "C": 12.011,
         "N": 14.007,
         "O": 15.999,
         "F": 18.99840316,
         "Ne": 20.18,
         "Na": 22.9897693,
         "Mg": 24.305,
         "Al": 26.981538,
         "Si": 28.085,
         "P": 30.973762,
         "S": 32.07,
         "Cl": 35.45,
         "Ar": 39.9,
         "K": 39.0983,
         "Ca": 40.08,
         "Sc": 44.95591,
         "Ti": 47.867,
         "V": 50.9415,
         "Cr": 51.996,
         "Mn": 54.93804,
         "Fe": 55.84,
         "Co": 58.93319,
         "Ni": 58.693,
         "Cu": 63.55,
         "Zn": 65.4,
         "Ga": 69.723,
         "Ge": 72.63,
         "As": 74.92159,
         "Se": 78.97,
         "Br": 79.9,
         "Kr": 83.8,
         "Rb": 85.468,
         "Sr": 87.62,
         "Y": 88.90584,
         "Zr": 91.22,
         "Nb": 92.90637,
         "Mo": 95.95,
         "Tc": 96.90636,
         "Ru": 101.1,
         "Rh": 102.9055,
         "Pd": 106.42,
         "Ag": 107.868,
         "Cd": 112.41,
         "In": 114.818,
         "Sn": 118.71,
         "Sb": 121.76,
         "Te": 127.6,
         "I": 126.9045,
         "Xe": 131.29,
         "Cs": 132.905452,
         "Ba": 137.33,
         "La": 138.9055,
         "Ce": 140.116,
         "Pr": 140.90766,
         "Nd": 144.24,
         "Pm": 144.91276,
         "Sm": 150.4,
         "Eu": 151.964,
         "Gd": 157.25,
         "Tb": 158.92535,
         "Dy": 162.5,
         "Ho": 164.93033,
         "Er": 167.26,
         "Tm": 168.93422,
         "Yb": 173.05,
         "Lu": 174.9667,
         "Hf": 178.49,
         "Ta": 180.9479,
         "W": 183.84,
         "Re": 186.207,
         "Os": 190.2,
         "Ir": 192.22,
         "Pt": 195.08,
         "Au": 196.96657,
         "Hg": 200.59,
         "Tl": 204.383,
         "Pb": 207.0,
         "Bi": 208.9804,
         "Po": 208.98243,
         "At": 209.98715,
         "Rn": 222.01758,
         "Fr": 223.01973,
         "Ra": 226.02541,
         "Ac": 227.02775,
         "Th": 232.038,
         "Pa": 231.03588,
         "U": 238.0289,
         "Np": 237.048172,
         "Pu": 244.0642,
         "Am": 243.06138,
         "Cm": 247.07035,
         "Bk": 247.07031,
         "Cf": 251.07959,
         "Es": 252.083,
         "Fm": 257.09511,
         "Md": 258.09843,
         "No": 259.101,
         "Lr": 266.12,
         "Rf": 267.122,
         "Db": 268.126,
         "Sg": 269.128,
         "Bh": 270.133,
         "Hs": 269.1336,
         "Mt": 277.154,
         "Ds": 282.166,
         "Rg": 282.169,
         "Cn": 286.179,
         "Nh": 286.182,
         "Fl": 290.192,
         "Mc": 290.196,
         "Lv": 293.205,
         "Ts": 294.211,
         "Og": 295.216}

#
# Levi-Civita symbols (2, 3, 4D)
#
levi_civita2 = np.zeros((2, 2), dtype=np.float64)
levi_civita2[0, 1] = 1
levi_civita2[1, 0] = -1
levi_civita3 = np.zeros((3, 3, 3), dtype=np.float64)
levi_civita3[0, 1, 2] = 1
levi_civita3[1, 2, 0] = 1
levi_civita3[2, 0, 1] = 1
levi_civita3[1, 0, 2] = -1
levi_civita3[2, 1, 0] = -1
levi_civita3[0, 2, 1] = -1

levi_civita4 = np.zeros((4, 4, 4, 4), dtype=np.float64)
levi_civita4[0, 1, 2, 3] = 1
levi_civita4[3, 0, 1, 2] = -1
levi_civita4[2, 3, 0, 1] = 1
levi_civita4[1, 2, 3, 0] = -1

levi_civita4[0, 1, 3, 2] = -1
levi_civita4[2, 0, 1, 3] = 1
levi_civita4[3, 2, 0, 1] = -1
levi_civita4[1, 3, 2, 0] = 1

levi_civita4[0, 2, 1, 3] = -1
levi_civita4[3, 0, 2, 1] = 1
levi_civita4[1, 3, 0, 2] = -1
levi_civita4[2, 1, 3, 0] = 1

levi_civita4[0, 2, 3, 1] = 1
levi_civita4[1, 0, 2, 3] = -1
levi_civita4[3, 1, 0, 2] = 1
levi_civita4[2, 3, 1, 0] = -1

levi_civita4[0, 3, 1, 2] = 1
levi_civita4[2, 0, 3, 1] = -1
levi_civita4[1, 2, 0, 3] = 1
levi_civita4[3, 1, 2, 0] = -1

levi_civita4[0, 3, 2, 1] = -1
levi_civita4[1, 0, 3, 2] = 1
levi_civita4[2, 1, 0, 3] = -1
levi_civita4[3, 2, 1, 0] = 1
