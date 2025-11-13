from PMNS import *
from hamiltonian_vs_packages import (
    P_nuexact_matter,
    P_nuexact_vacuum,
    P_nufast_matter,
    P_nufast_vacuum,
    P_hamiltonian_expm,
    P_hamiltonian_eigh,
)
import matplotlib.pyplot as plt

# Plot of oscillation probability from different packages in an energy range
plt.figure(figsize=(12,5))
plt.plot(E_range, P_nufast_matter, label=r'$P(\nu_\mu \to \nu_e) (NuFast\_matter)$', color='yellow', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(E_range, P_nufast_vacuum, label=r'$P(\nu_\mu \to \nu_e) (NuFast\_vacuum)$', color='yellow', linestyle=':', linewidth=2, alpha= 0.8)
plt.plot(E_range, P_nuexact_matter, label=r'$P(\nu_\mu \to \nu_e) (NuExact\_matter)$', color='red', linestyle='-', linewidth=2, alpha=0.7)
plt.plot(E_range, P_nuexact_vacuum, label=r'$P(\nu_\mu \to \nu_e) (NuExact\_vacuum)$', color='red', linestyle=':', linewidth=2, alpha=0.7)
plt.plot(E_range, P_hamiltonian_expm, label=r'$P(\nu_\mu \to \nu_e) (hamiltonian\_expm)$', color='blue', linestyle='-', linewidth=2, alpha=0.9)
plt.plot(E_range, P_hamiltonian_eigh, label=r'$P(\nu_\mu \to \nu_e) (hamiltonian\_eigh)$', color='blue', linestyle=':', linewidth=2, alpha=0.9)
#plt.plot(E_range, Pee_ana, label=r'$P(\nu_e \to \nu_e) (Analytical)$', color='blue', linestyle='-', linewidth=2)
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel('Probability')
plt.title(f'P(numu2nue) at L={L} km ({exp})')
plt.grid(True)
plt.ylim(0, 1.1)

# Annotate the maximum (minimum survival)
#max_osc_idx = np.argmax(Pmu_tau)
#E_max_osc = E_range[max_osc_idx]
#plt.axvline(x=E_max_osc, color='gray', linestyle='--', label=f'E ~ {E_max_osc:.2f} GeV')

plt.legend()
plt.tight_layout()
plt.show()

# Plot the oscillation probability within a considered energy
#E0 = 2.5   # GeV, representative energy
#Pmu_e_3 = np.round(P_3nu_matter_evolutor(L, E0, U, [delta_m21, delta_m31], rho, 1, 0), 5)
#Pmu_tau_3 = np.round(P_3nu_matter_evolutor(L, E0, U, [delta_m21, delta_m31], rho, 1, 2), 5)
#Pmu_mu_3 = np.round(P_3nu_matter_evolutor(L, E0, U, [delta_m21, delta_m31], rho, 1, 1), 5)

#print(f"At L={L} km, E={E0} GeV, rho={rho} g/cm^3:")
#print(f"P(\nu_\mu -> \nu_e)  = {Pmu_e_3}")
#print(f"P(\nu_\mu -> \nu_\tau)  = {Pmu_tau_3}")
#print(f"P(\nu_\mu -> \nu_\mu)  = {Pmu_mu_3}")
#print("Total Probability: ", Pmu_e_3 + Pmu_tau_3 + Pmu_mu_3)
#print(f"Total probability deviation: {Pmu_e_3 + Pmu_tau_3 + Pmu_mu_3 - 1:.5e}") 

# Absolute difference
y1 = P_nuexact_matter   
y2 = P_nufast_matter
diff = y1 - y2
plt.figure(figsize=(12,5))
plt.title(f'eigh vs NuFAST at L={L} km ({exp}) in matter')
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel(r'$\Delta P(\nu_\mu \to \nu_e)$')
plt.plot(E_range, diff, label='Difference')
plt.grid(True)
plt.tight_layout()
plt.show()