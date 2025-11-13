"""
Parameters:
    L : float
        Baseline in km
    E : float
        Neutrino energy in GeV
    U : 3x3 complex ndarray
        PMNS mixing matrix
"""
import numpy as np

# Experiment's parameters
exp = "DUNE"
L = 1285                               # km
E_range = np.linspace(0.1, 3, 1000)     # GeV
# Matter parameters
G_F = 1.1663787e-5     # GeV^-2
rho = 2.825            # g/cm^3
Ye = 0.5
Na = 6.02214076e23     # mol^-1
# PMNS parameters (NuFIT 6.0 best fit, NH/IH)
theta12 = np.radians(33.68)
theta23 = np.radians(48.6)
theta13 = np.radians(8.58)
delta_cp = np.radians(285)
s12, c12 = np.sin(theta12), np.cos(theta12)
s23, c23 = np.sin(theta23), np.cos(theta23)
s13, c13 = np.sin(theta13), np.cos(theta13)
# Mass-squared differences (NuFIT 6.0 best fit, NH/IH)
Dmsq21 = 7.49e-5  # eV^2
Dmsq31 = -2.510e-3 # eV^2  

# PMNS matrix
U = np.array([
    [ c12*c13,               s12*c13,            s13*np.exp(-1j*delta_cp)],
    [-s12*c23 - c12*s23*s13*np.exp(1j*delta_cp),  c12*c23 - s12*s23*s13*np.exp(1j*delta_cp), s23*c13 ],
    [ s12*s23 - c12*c23*s13*np.exp(1j*delta_cp), -c12*s23 - s12*c23*s13*np.exp(1j*delta_cp), c23*c13 ]
], dtype=complex)

# Unitarity test
I = np.eye(U.shape[0])
norm = np.linalg.norm(U.conj().T @ U - I, "fro")
print("U\u2020U - I:", norm)