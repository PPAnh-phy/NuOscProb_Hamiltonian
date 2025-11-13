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
exp = "MINOS"
L = 735                               # km
E_range = np.linspace(0.1, 10, 500)    # GeV

# PMNS parameters (NuFIT 6.0, NH)
theta12 = np.radians(33.68)
theta23 = np.radians(48.5)
theta13 = np.radians(8.52)
delta_cp = 0

s12, c12 = np.sin(theta12), np.cos(theta12)
s23, c23 = np.sin(theta23), np.cos(theta23)
s13, c13 = np.sin(theta13), np.cos(theta13)

# PMNS matrix
U = np.array([
    [ c12*c13,               s12*c13,            s13*np.exp(-1j*delta_cp)],
    [-s12*c23 - c12*s23*s13*np.exp(1j*delta_cp),  c12*c23 - s12*s23*s13*np.exp(1j*delta_cp), s23*c13 ],
    [ s12*s23 - c12*c23*s13*np.exp(1j*delta_cp), -c12*s23 - s12*c23*s13*np.exp(1j*delta_cp), c23*c13 ]
], dtype=complex)

# Mass-squared differences (NuFIT 6.0, NH)
delta_m21 = 7.49e-5
delta_m31 = 2.534e-3

# 3nu_vacuum_analytical_NH [2]
def P_3nu_vacuum_anal(L, E, U, delta_m2, alpha, beta):
  hbarc = 197.3269804e-9     # eV.m
  L_natural_unit = (L * 1e3) / hbarc
  E_eV = E * 1e9   # ev
  dm21, dm31 = delta_m2
  dm32 = dm31 - dm21
  dm2 = [0, dm21, dm31]  # m1^2=0 reference
  prob = 0.0
  for i in range(3):
    for j in range(i):
      dm2_ij = dm2[i] - dm2[j]
      phase = dm2_ij * L_natural_unit / (4*E_eV)
      U_alpha_i = U[alpha, i]
      U_beta_i  = U[beta, i]
      U_alpha_j = U[alpha, j]
      U_beta_j  = U[beta, j]
      real_part = np.real(U_alpha_i * np.conj(U_beta_i) * np.conj(U_alpha_j) * U_beta_j)
      imag_part = np.imag(U_alpha_i * np.conj(U_beta_i) * np.conj(U_alpha_j) * U_beta_j)
      prob -= 4 * real_part * np.sin(phase)**2
      prob += 2 * imag_part * np.sin(2*phase)
  if alpha == beta:
    prob = 1 + prob

  return prob

# 3nu_vacuum_evolutor_NH
def P_3nu_vacuum_evolutor(L, E, U, delta_m2, alpha, beta):
    hbarc = 197.3269804e-9     # eV.m
    L_natural_unit = (L * 1e3) / hbarc   
    E_eV = E * 1e9   # eV
    dm21, dm31 = delta_m2
    dm32 = dm31 - dm21
    dm2 = [0, dm21, dm31]  # m1^2=0 reference
    # Hamiltonian in mass basis    
    H_mass = np.diag([0, dm21/(2*E_eV), dm31/(2*E_eV)])
    # Hamiltonian in flavor basis
    H_flavor = U @ H_mass @ U.conj().T
    # Time evolution operator
    U_t = expm(-1j * H_flavor * L_natural_unit)
    # Initial and final flavor states
    psi_alpha = np.zeros(3, dtype=complex)
    psi_alpha[alpha] = 1
    psi_beta = np.zeros(3, dtype=complex)
    psi_beta[beta] = 1
    # Evolve
    psi_t = U_t @ psi_alpha
    amplitude = np.vdot(psi_beta, psi_t)
    probability = np.abs(amplitude)**2

    return probability


# Compute probabilities over E (analytical)
P_mue = np.array([P_3nu_vacuum_anal(L, E, U, [delta_m21, delta_m31], 1, 0) for E in E_range])
P_mutau = np.array([P_3nu_vacuum_anal(L, E, U, [delta_m21, delta_m31], 1, 2) for E in E_range])
P_mumu = np.array([P_3nu_vacuum_anal(L, E, U, [delta_m21, delta_m31], 1, 1) for E in E_range])
# Compute probabilities over E (exact)
Pmue = np.array([P_3nu_vacuum_evolutor(L, E, U, [delta_m21, delta_m31], 1, 0) for E in E_range])
Pmutau = np.array([P_3nu_vacuum_evolutor(L, E, U, [delta_m21, delta_m31], 1, 2) for E in E_range])
Pmumu = np.array([P_3nu_vacuum_evolutor(L, E, U, [delta_m21, delta_m31], 1, 1) for E in E_range])

# Plotting (analytical)
plt.figure(figsize=(12,5))
plt.plot(E_range, P_mue, label=r'$P(\nu_\mu \to \nu_e)$', color='green')
plt.plot(E_range, P_mutau, label=r'$P(\nu_\mu \to \nu_\tau)$', color='red')
plt.plot(E_range, P_mumu, label=r'$P(\nu_\mu \to \nu_\mu)$', color='blue')
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel('Probability')
plt.title(f'Neutrino Oscillation Probabilities at L={L} km ({exp}) — Analytical')
plt.grid(True)
plt.ylim(0, 1.1)
# Annotate the maximum (minimum survival)
max_osc_idx = np.argmax(P_mutau)
E_max_osc = E_range[max_osc_idx]
plt.axvline(x=E_max_osc, color='gray', linestyle='--', label=f'E ~ {E_max_osc:.2f} GeV')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting (hamiltonian)
plt.figure(figsize=(12,5))
plt.plot(E_range, Pmue, label=r'$P(\nu_\mu \to \nu_e)$', color='green')
plt.plot(E_range, Pmutau, label=r'$P(\nu_\mu \to \nu_\tau)$', color='red')
plt.plot(E_range, Pmumu, label=r'$P(\nu_\mu \to \nu_\mu)$', color='blue')
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel('Probability')
plt.title(f'Neutrino Oscillation Probabilities at L={L} km ({exp}) — Time Evolution')
plt.grid(True)
plt.ylim(0, 1.1)
# Annotate the maximum (minimum survival)
max_osc_idx = np.argmax(Pmutau)
E_max_osc = E_range[max_osc_idx]
plt.axvline(x=E_max_osc, color='gray', linestyle='--', label=f'E ~ {E_max_osc:.2f} GeV')
plt.legend()
plt.tight_layout()
plt.show()

E0 = 3   # GeV

P_mue_3 = np.round(P_3nu_vacuum_anal(L, E0, U, [delta_m21, delta_m31], 1, 0), 5)
P_mutau_3 = np.round(P_3nu_vacuum_anal(L, E0, U, [delta_m21, delta_m31], 1, 2), 5)
P_mumu_3 = np.round(P_3nu_vacuum_anal(L, E0, U, [delta_m21, delta_m31], 1, 1), 5)
print(f"At L={L} km, E={E0} GeV, delta CP={delta_cp} (analytical formula):")
print(f"P(v_mu -> v_e)  = {P_mue_3}")
print(f"P(ν_mu -> ν_tau)  = {P_mutau_3}")
print(f"P(ν_mu -> ν_mu)  = {P_mumu_3}")
print("Total Probability: ", P_mue_3 + P_mutau_3 + P_mumu_3)
print(f"Total probability deviation: {P_mue_3 + P_mutau_3 + P_mumu_3 - 1:.5e}")

Pmue_3 = np.round(P_3nu_vacuum_evolutor(L, E0, U, [delta_m21, delta_m31], 1, 0), 5)
Pmutau_3 = np.round(P_3nu_vacuum_evolutor(L, E0, U, [delta_m21, delta_m31], 1, 2), 5)
Pmumu_3 = np.round(P_3nu_vacuum_evolutor(L, E0, U, [delta_m21, delta_m31], 1, 1), 5)
print(f"At L={L} km, E={E0} GeV, delta CP={delta_cp} (hamiltonian):")
print(f"P(ν_mu -> ν_e)  = {Pmue_3}")
print(f"P(ν_mu -> ν_tau)  = {Pmutau_3}")
print(f"P(ν_mu -> ν_mu)  = {Pmumu_3}")
print("Total Probability: ", Pmue_3 + Pmutau_3 + Pmumu_3)
print(f"Total Probability Deviation: {Pmue_3 + Pmutau_3 + Pmumu_3 - 1:.5e}")

# Absolute difference
y1 = Pmue     # hamiltonian
y2 = P_mue    # analytic
# Difference plot
abs_diff = y1 - y2
plt.figure(figsize=(12,5))
plt.title(f'Analytical Formula vs Hamiltonian at L={L} km ({exp}) in vacuum')
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel(r'$\Delta P(\nu_\mu \to \nu_e)$')
plt.plot(E_range, abs_diff, label='Difference')
plt.grid(True)
plt.show()

import time
# -----------------------------------
# User parameters
# ----------------------------------- 
N_eval = len(E_range)       
N_repeat = 50                 # number of repeated runs for statistics
methods = ["analytical", "hamiltonian"]

times_anal = []
times_hal = []
# -----------------------------------
# Benchmark loop
# -----------------------------------
for i in range(N_repeat):
    # Analytical formula
    t0 = time.time()
    for E in E_range:
        P_3nu_vacuum_anal(L, E0, U, [delta_m21, delta_m31], 1, 0)
    t1 = time.time()
    dt = t1 - t0
    times_anal.append(dt)

    # Hamiltonian (expm)
    t0 = time.time()
    for E in E_range:
        P_3nu_vacuum_evolutor(L, E, U, [delta_m21, delta_m31], 1, 0)
    t1 = time.time()
    dt = t1 - t0
    times_hal.append(dt)
    
# -----------------------------------
# Statistics
# -----------------------------------
def summarize(times):
    return np.mean(times), np.std(times), np.min(times), np.max(times)

mean_anal, std_anal, min_anal, max_anal = summarize(times_anal)
mean_hal, std_hal, min_hal, max_hal = summarize(times_hal)

print("\n===== Summary Statistics =====")
print(f"Analytical Formula:  mean={mean_anal:.4f}s ±{std_anal:.4f}  [min={min_anal:.4f}, max={max_anal:.4f}]")
print(f"Hamiltonian:  mean={mean_hal:.4f}s ±{std_hal:.4f}  [min={min_hal:.4f}, max={max_hal:.4f}]")

# -----------------------------------
# Visualization
# -----------------------------------
means = [mean_anal, mean_hal]
stds = [std_anal, std_hal]

plt.figure(figsize=(8,5))
plt.bar(methods, means, yerr=stds, capsize=6, color=["#17becf", "#2ca02c"])
plt.ylabel("Total runtime [s]")
plt.title(f"Average runtime for {N_eval} evaluations (over {N_repeat} runs)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Overall performance
N = 10000   # number of evaluations 
E0 = 2.5   # GeV, representative energy

print("\nThe running time of:")

# analytical formula
t0 = time.time()
for _ in range(N):
    P_3nu_vacuum_anal(L, E0, U, [delta_m21, delta_m31], 1, 0)
t1 = time.time()
print(f"Analytical Formula: {t1 - t0:.4f} s for {N} evals")

# hamiltonian
t0 = time.time()
for _ in range(N):
    P_3nu_vacuum_evolutor(L, E, U, [delta_m21, delta_m31], 1, 0)
t1 = time.time()
print(f"Hamiltonian (expm): {t1 - t0:.4f} s for {N} evals")