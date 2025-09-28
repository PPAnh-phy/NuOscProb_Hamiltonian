"""
Parameters:
    L : float
        Baseline in km
    E : float
        Neutrino energy in GeV
    U : 3x3 complex ndarray
        PMNS mixing matrix
    delta_m2 : list or array of float
        Mass-squared differences in eV^2
    alpha : int
        Initial flavor (0=e, 1=mu, 2=tau)
    beta : int
        Final flavor (0=e, 1=mu, 2=tau)

Returns:
    P : float
        Oscillation probability
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Experiment's parameters
exp = "DUNE"   # For example. One can change these parameters in need.
L = 1285                               # km
E_range = np.linspace(0.1, 10, 500)     # GeV
# Matter parameters
rho = 2.825            # g/cm^3
Ye = 0.5               # electron fraction of the Earth's crust
# PMNS parameters (NuFIT 5.1, NH)
theta12 = np.radians(33.45)
theta23 = np.radians(42.1)
theta13 = np.radians(8.62)
delta_cp = 0.7 * np.pi
s12, c12 = np.sin(theta12), np.cos(theta12)
s23, c23 = np.sin(theta23), np.cos(theta23)
s13, c13 = np.sin(theta13), np.cos(theta13)
# Mass-squared differences (NuFIT 5.1, NH)
Dmsq21 = 7.42e-5  # eV^2
Dmsq31 = 2.515e-3 # eV^2

# PMNS matrix
U = np.array([
    [ c12*c13,               s12*c13,            s13*np.exp(-1j*delta_cp)],
    [-s12*c23 - c12*s23*s13*np.exp(1j*delta_cp),  c12*c23 - s12*s23*s13*np.exp(1j*delta_cp), s23*c13 ],
    [ s12*s23 - c12*c23*s13*np.exp(1j*delta_cp), -c12*s23 - s12*c23*s13*np.exp(1j*delta_cp), c23*c13 ]
], dtype=complex)

# expm
def P_3nu_evolutor_expm(L, E, U, delta_m2, rho, alpha, beta): 
	L_natural_unit = L * 5.067730718e9   # # km -> 1e3 m -> 1e3/hbarc eV^-1
	E_eV = E * 1e9   # eV
	# Normal Ordering
	dm21, dm31 = delta_m2
	dm32 = dm31 - dm21
	dm2 = [0, dm21, dm31]  # m1^2=0 reference
	# Inverted ordering
	#dm21, dm31 = delta_m2  
	#m2sq = dm31 + dm21   
	#dm2 = [dm31, m2sq, 0]  # m3^2 = 0 reference
	# Hamiltonian
	H_mass = np.diag([0, dm21/(2*E_eV), dm31/(2*E_eV)])
	H_flavor = U @ H_mass @ U.conj().T
    # Additional potential term
	V = np.sqrt(2) * G_F * rho * Ye * Na * 7.645373e-33 # eV
	#V = np.sqrt(2) * rho * Ye * 5.370183934e-14 # GF*Na*7.645373e-33eV
	V_matter = np.diag([V, 0, 0])
	H_flavor += V_matter
    # Time evolution
	U_t = expm(-1j * H_flavor * L_natural_unit)
    # Flavor states
	psi_alpha = np.zeros(3, dtype=complex)
	psi_alpha[alpha] = 1
	psi_beta = np.zeros(3, dtype=complex)
	psi_beta[beta] = 1
    # Evolve and calculate probability
	psi_t = U_t @ psi_alpha
	amplitude = np.vdot(psi_beta, psi_t)
	probability = np.abs(amplitude)**2
	
	return probability

# Eigh
def P_3nu_evolutor_eigh(L, E, U, delta_m2, rho, alpha, beta): 
	hbarc = 197.3269804e-9     # eV.m
	L_natural_unit = (L * 1e3) / hbarc   # eV^-1
	#L_natural_unit = L * 5.067730718e9   # km -> 1e3 m -> 1e3/hbarc eV^-1
	E_eV = E * 1e9   # eV
	# Normal Ordering
	dm21, dm31 = delta_m2
	dm32 = dm31 - dm21
	dm2 = [0, dm21, dm31]  # m1^2=0 reference
	# Inverted Ordering
	#dm21, dm31 = delta_m2  
	#m2sq = dm31 + dm21   
	#dm2 = [dm31, m2sq, 0]  # m3^2 = 0 reference
	# Hamiltonian
	H_mass = np.diag([0, dm21/(2*E_eV), dm31/(2*E_eV)])
	H_flavor = U @ H_mass @ U.conj().T
    # Additional potential term
	V = np.sqrt(2) * G_F * rho * Ye * Na * 7.645373e-33 # eV
	#V = np.sqrt(2) * rho * Ye * 5.370183934e-14 # GF*Na*7.645373e-33eV
	V_matter = np.diag([V, 0, 0])
	H_flavor += V_matter
    # Eigen-decomposition
	evals, evecs = np.linalg.eigh(H_flavor)
    # Time evolution 
	phases = np.exp(-1j * evals * L_natural_unit)
	U_t = evecs @ np.diag(phases) @ evecs.conj().T
    # Initial and final flavor states
	psi_alpha = np.zeros(3, dtype=complex)
	psi_alpha[alpha] = 1
	psi_beta = np.zeros(3, dtype=complex)
	psi_beta[beta] = 1
    # Probability
	psi_t = U_t @ psi_alpha
	amplitude = np.vdot(psi_beta, psi_t)
	probability = np.abs(amplitude)**2
	
	return probability

# Unitarity test
I = np.eye(U.shape[0])
norm = np.linalg.norm(U.conj().T @ U - I, "fro")
print(r'$U^\dagger U - I$: ', norm)

# Running time investigation
import time

times_expm = []
times_eigh = []

for E in E_range:
    # Hamiltonian_expm
    t0 = time.time()
    P_3nu_evolutor_expm(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0)
    times_expm.append(time.time() - t0)
    # Hamiltonian_eigh
    t0 = time.time()
    P_3nu_evolutor_eigh(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0)
    times_eigh.append(time.time() - t0)

times_expm = np.array(times_expm)
times_eigh = np.array(times_eigh)