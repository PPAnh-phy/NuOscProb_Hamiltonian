"""
Parameters:
    delta_m2 : list or array of float
        Mass-squared differences in eV^2
    hierarchy: "normal" or "inverted"
    alpha : int
        Initial flavor (0=e, 1=mu, 2=tau)
    beta : int
        Final flavor (0=e, 1=mu, 2=tau)
Returns:
    P : float
        Oscillation probability
"""

from PMNS import *

from scipy.linalg import expm, eigh

def mass_splittings(delta_m2, hierarchy="inverted"):
    dm21, dm31 = delta_m2
    if hierarchy == "normal":
        dm32 = dm31 - dm21
        dm2 = [0, dm21, dm31]  # m1^2 = 0 reference
    elif hierarchy == "inverted":
        dm31 = -abs(dm31)      # Here dm31 is NEGATIVE (Dmsq31 = -2.484e-3 eV^2)
        dm32 = dm31 + dm21     # shift by solar splitting
        dm2 = [dm31, dm32, 0]  # m3^2 = 0 reference
    else:
        raise ValueError("Hierarchy must be 'normal' or 'inverted'.")
    return dm2

def P_3nu_evolutor_expm(L, E, U, delta_m2, rho, alpha, beta, hierarchy="inverted"):
    # natural unit conversion
    hbarc = 197.3269804e-9     # eV.m
    L_natural_unit = (L * 1e3) / hbarc   # eV^-1
    E_eV = E * 1e9   # eV
    # dm2 from chosen mass hierachy
    dm2 = mass_splittings(delta_m2, hierarchy)
    # Hamiltonian
    H_mass = np.diag([dm2[0]/(2*E_eV), dm2[1]/(2*E_eV), dm2[2]/(2*E_eV)])
    H_flavor = U @ H_mass @ U.conj().T
    # Additional potential term
    V = np.sqrt(2) * G_F * rho * Ye * Na * 7.645373e-33 # eV
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

def P_3nu_evolutor_eigh(L, E, U, delta_m2, rho, alpha, beta, hierarchy="inverted"):
    # natural unit conversion
    hbarc = 197.3269804e-9     # eV.m
    L_natural_unit = (L * 1e3) / hbarc   # eV^-1
    E_eV = E * 1e9   # eV
    # dm2 from chosen mass hierachy
    dm2 = mass_splittings(delta_m2, hierarchy)
    # Hamiltonian
    H_mass = np.diag([dm2[0]/(2*E_eV), dm2[1]/(2*E_eV), dm2[2]/(2*E_eV)])
    H_flavor = U @ H_mass @ U.conj().T
    # Additional potential term
    V = np.sqrt(2) * G_F * rho * Ye * Na * 7.645373e-33 # eV
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

# NuFast
eVsqkm_to_GeV_over4 = (1e3 / 197.3269804e-9) / (4 * 1e9)  # My modification
YerhoE2a = 1.51891739e-4  # My modification
#eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
#YerhoE2a = 1.52e-4
# --------------------------------------------------------------------- #
# Set the number of Newton-Raphson iterations which sets the precision. #
# 0 is close to the single precision limit and is better than DUNE/HK   #
# in the high statistics regime. Increasing N_Newton to 1,2,... rapidly #
# improves the precision at a modest computational cost                 #
# --------------------------------------------------------------------- #
N_Newton = 0

#s12sq = 0.30
#s13sq = 0.022
#s23sq = 0.47
s12sq = np.sin(theta12)**2  # My modification
s13sq = np.sin(theta13)**2  # My modification
s23sq = np.sin(theta23)**2  # My modification

### NUFAST
def Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_cp, Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton):
    # --------------------------------------------------------------------- #
    # First calculate useful simple functions of the oscillation parameters #
    # --------------------------------------------------------------------- #
    c13sq = 1 - s13sq

    # Ueisq's
    Ue2sq = c13sq * s12sq
    Ue3sq = s13sq

    # Umisq's, Utisq's and Jvac
    Um3sq = c13sq * s23sq
    # Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
    Ut2sq = s13sq * s12sq * s23sq
    Um2sq = (1 - s12sq) * (1 - s23sq)

    Jrr = np.sqrt(Um2sq * Ut2sq)
    sind = np.sin(delta_cp)
    cosd = np.cos(delta_cp)

    Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
    Jmatter = 8 * Jrr * c13sq * sind
    Amatter = Ye * rho * E * YerhoE2a
    Dmsqee = Dmsq31 - s12sq * Dmsq21

    # calculate A, B, C, See, Tee, and part of Tmm
    A = Dmsq21 + Dmsq31 # temporary variable
    See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq
    Tmm = Dmsq21 * Dmsq31 # using Tmm as a temporary variable
    Tee = Tmm * (1 - Ue3sq - Ue2sq)
    C = Amatter * Tee
    A = A + Amatter

    # ---------------------------------- #
    # Get lambda3 from lambda+ of MP/DMP #
    # ---------------------------------- #
    xmat = Amatter / Dmsqee
    tmp = 1 - xmat
    lambda3 = Dmsq31 + 0.5 * Dmsqee * (xmat - 1 + np.sqrt(tmp * tmp + 4 * s13sq * xmat))

    # ---------------------------------------------------------------------------- #
    # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
    # ---------------------------------------------------------------------------- #
    B = Tmm + Amatter * See # B is only needed for N_Newton >= 1
    for i in range(N_Newton):
        lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * (2 * (lambda3 - A) + lambda3) + B) # this strange form prefers additions to multiplications

    # ------------------- #
    # Get  Delta lambda's #
    # ------------------- #
    tmp = A - lambda3
    Dlambda21 = np.sqrt(tmp * tmp - 4 * C / lambda3)
    lambda2 = 0.5 * (A - lambda3 + Dlambda21)
    Dlambda32 = lambda3 - lambda2
    Dlambda31 = Dlambda32 + Dlambda21

    # ----------------------- #
    # Use Rosetta for Veisq's #
    # ----------------------- #
    # denominators
    PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21)
    Xp3 = PiDlambdaInv * Dlambda21
    Xp2 = -PiDlambdaInv * Dlambda31

    # numerators
    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

    Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq
    Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A)

    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

    # ------------- #
    # Use NHS for J #
    # ------------- #
    Jmatter = Jmatter * Dmsq21 * Dmsq31 * (Dmsq31 - Dmsq21) * PiDlambdaInv

    # ----------------------- #
    # Get all elements of Usq #
    # ----------------------- #
    Ue1sq = 1 - Ue3sq - Ue2sq
    Um1sq = 1 - Um3sq - Um2sq

    Ut3sq = 1 - Um3sq - Ue3sq
    Ut2sq = 1 - Um2sq - Ue2sq
    Ut1sq = 1 - Um1sq - Ue1sq

    # ----------------------- #
    # Get the kinematic terms #
    # ----------------------- #
    Lover4E = eVsqkm_to_GeV_over4 * L / E

    D21 = Dlambda21 * Lover4E
    D32 = Dlambda32 * Lover4E

    sinD21 = np.sin(D21)
    sinD31 = np.sin(D32 + D21)
    sinD32 = np.sin(D32)

    triple_sin = sinD21 * sinD31 * sinD32

    sinsqD21_2 = 2 * sinD21 * sinD21
    sinsqD31_2 = 2 * sinD31 * sinD31
    sinsqD32_2 = 2 * sinD32 * sinD32

    # ------------------------------------------------------------------- #
    # Calculate the three necessary probabilities, separating CPC and CPV #
    # ------------------------------------------------------------------- #
    Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 \
                + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 \
                + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
    Pme_CPV = -Jmatter * triple_sin

    Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2 \
                 + Um3sq * Um1sq * sinsqD31_2 \
                 + Um3sq * Um2sq * sinsqD32_2)

    Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 \
                 + Ue3sq * Ue1sq * sinsqD31_2 \
                 + Ue3sq * Ue2sq * sinsqD32_2)

    # ---------------------------- #
    # Assign all the probabilities #
    # ---------------------------- #
    #probs_returned = np.empty((3, 3))
    #probs_returned[0][0] = Pee												# Pee
    #probs_returned[0][1] = Pme_CPC - Pme_CPV								# Pem
    #probs_returned[0][2] = 1 - Pee - probs_returned[0][1]  			    # Pet

    #probs_returned[1][0] = Pme_CPC + Pme_CPV								# Pme
    #probs_returned[1][1] = Pmm												# Pmm
    #probs_returned[1][2] = 1 - probs_returned[1][0] - Pmm					# Pmt

    #probs_returned[2][0] = 1 - Pee - probs_returned[1][0]					# Pte
    #probs_returned[2][1] = 1 - probs_returned[0][1] - Pmm					# Ptm
    #probs_returned[2][2] = 1 - probs_returned[0][2] - probs_returned[1][2]	# Ptt

    probs_ee = Pee
    probs_em = Pme_CPC - Pme_CPV
    probs_et = 1 - Pee - probs_em

    probs_me = Pme_CPC + Pme_CPV
    probs_mm = Pmm
    probs_mt = 1 - probs_me - Pmm

    return probs_me    

def Probability_Vacuum_LBL(s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31, L, E):
    # --------------------------------------------------------------------- #
    # First calculate useful simple functions of the oscillation parameters #
    # --------------------------------------------------------------------- #
    c13sq = 1 - s13sq

    # Ueisq's
    Ue3sq = s13sq
    Ue2sq = c13sq * s12sq

    # Umisq's, Utisq's and Jvac
    Um3sq = c13sq * s23sq
    # Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
    Ut2sq = s13sq * s12sq * s23sq
    Um2sq = (1 - s12sq) * (1 - s23sq)

    Jrr = np.sqrt(Um2sq * Ut2sq)
    sind = np.sin(delta)
    cosd = np.cos(delta)
    Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
    Jvac = 8 * Jrr * c13sq * sind

    # ----------------------- #
    # Get all elements of Usq #
    # ----------------------- #
    Ue1sq = 1 - Ue3sq - Ue2sq
    Um1sq = 1 - Um3sq - Um2sq

    Ut3sq = 1 - Um3sq - Ue3sq
    Ut2sq = 1 - Um2sq - Ue2sq
    Ut1sq = 1 - Um1sq - Ue1sq

    # ----------------------- #
    # Get the kinematic terms #
    # ----------------------- #
    Lover4E = eVsqkm_to_GeV_over4 * L / E

    D21 = Dmsq21 * Lover4E
    D31 = Dmsq31 * Lover4E

    sinD21 = np.sin(D21)
    sinD31 = np.sin(D31)
    sinD32 = np.sin(D31-D21)

    triple_sin = sinD21 * sinD31 * sinD32

    sinsqD21_2 = 2 * sinD21 * sinD21
    sinsqD31_2 = 2 * sinD31 * sinD31
    sinsqD32_2 = 2 * sinD32 * sinD32

    # ------------------------------------------------------------------- #
    # Calculate the three necessary probabilities, separating CPC and CPV #
    # ------------------------------------------------------------------- #
    Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 \
            + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 \
            + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2

    Pme_CPV = -Jvac * triple_sin

    Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2 \
                 + Um3sq * Um1sq * sinsqD31_2 \
                 + Um3sq * Um2sq * sinsqD32_2)

    Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 \
                 + Ue3sq * Ue1sq * sinsqD31_2 \
                 + Ue3sq * Ue2sq * sinsqD32_2)

    # ---------------------------- #
    # Assign all the probabilities #
    # ---------------------------- #
    #probs_returned = np.empty((3, 3))
    #probs_returned[0][0] = Pee												# Pee
    #probs_returned[0][1] = Pme_CPC - Pme_CPV								# Pem
    #probs_returned[0][2] = 1 - Pee - probs_returned[0][1]  	 			# Pet

    #probs_returned[1][0] = Pme_CPC + Pme_CPV								# Pme
    #probs_returned[1][1] = Pmm												# Pmm
    #probs_returned[1][2] = 1 - probs_returned[1][0] - Pmm					# Pmt

    #probs_returned[2][0] = 1 - Pee - probs_returned[1][0]					# Pte
    #probs_returned[2][1] = 1 - probs_returned[0][1] - Pmm					# Ptm
    #probs_returned[2][2] = 1 - probs_returned[0][2] - probs_returned[1][2]	# Ptt

    probs_ee = Pee
    probs_em = Pme_CPC - Pme_CPV
    probs_et = 1 - Pee - probs_em

    probs_me = Pme_CPC + Pme_CPV
    probs_mm = Pmm
    probs_mt = 1 - probs_me - Pmm

    return probs_me

### NuExact
import sys
sys.path.append('D:/Documents/USTH/USTH/Internships/IFIRSE - Neutrino Physics/Code/NuOscProbExact-master/NuOscProbExact-master/src')
import oscprob3nu
import hamiltonians3nu
from globaldefs import *

def oscprob_nuexact_vacuum(L, E, delta_m21_sq, delta_m31_sq, U):
    h_vacuum_energy_indep = \
    hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(  S12_IO_BF,
                                                                S23_IO_BF,
                                                                S13_IO_BF,
                                                                DCP_IO_BF,
                                                                D21_IO_BF,
                                                                D31_IO_BF)
    h_vacuum = np.multiply(1./(E*1e9), h_vacuum_energy_indep)

        # CONV_KM_TO_INV_EV is pulled from globaldefs; it converts km to eV^{-1}
    Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
            oscprob3nu.probabilities_3nu(h_vacuum, L*CONV_KM_TO_INV_EV)

    return Pme

def oscprob_nuexact_matter(L, E, delta_m21_sq, delta_m31_sq, U):
    h_vacuum_energy_indep = \
    hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(  S12_IO_BF,
                                                                S23_IO_BF,
                                                                S13_IO_BF,
                                                                DCP_IO_BF,
                                                                D21_IO_BF,
                                                                D31_IO_BF)

    # Units of VCC_EARTH_CRUST: [eV]
    h_matter = hamiltonians3nu.hamiltonian_3nu_matter(  h_vacuum_energy_indep,
                                                        E*1e9,
                                                        VCC_EARTH_CRUST)

    Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
        oscprob3nu.probabilities_3nu(h_matter, L*CONV_KM_TO_INV_EV)

    return Pme

# Analytical formula for Pee_JUNO
#def Pee_analytical(L, E, theta12, theta13, Dmsq21, Dmsq31):
    hbarc = 197.3269804e-9     # eV.m
    L_natural_unit = (L * 1e3) / hbarc
    E_eV = E * 1e9
    Dmsq32 = Dmsq31 - Dmsq21	
    s12, c12 = np.sin(theta12), np.cos(theta12)
    s13, c13 = np.sin(theta13), np.cos(theta13)   
    # oscillation phases
    phi21 = Dmsq21 * L_natural_unit / (4*E_eV) 
    phi31 = Dmsq31 * L_natural_unit / (4*E_eV) 
    phi32 = Dmsq32 * L_natural_unit / (4*E_eV)	
    term21 = (c13**4) * (np.sin(2*theta12)**2) * (np.sin(phi21)**2)
    term31 = (np.sin(2*theta13)**2) * (c12**2) * (np.sin(phi31)**2)
    term32 = (np.sin(2*theta13)**2) * (s12**2) * (np.sin(phi32)**2)
    
    return 1 - term21 - term31 - term32


P_hamiltonian_expm = np.array([P_3nu_evolutor_expm(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0) for E in E_range])
P_hamiltonian_eigh = np.array([P_3nu_evolutor_eigh(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0) for E in E_range])

P_nufast_matter = np.array([Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_cp, Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton) for E in E_range])
P_nufast_vacuum = np.array([Probability_Vacuum_LBL(s12sq, s13sq, s23sq, delta_cp, Dmsq21, Dmsq31, L, E) for E in E_range])

P_nuexact_matter = np.array([oscprob_nuexact_matter(L, E, Dmsq21, Dmsq31, U) for E in E_range])
P_nuexact_vacuum = np.array([oscprob_nuexact_vacuum(L, E, Dmsq21, Dmsq31, U) for E in E_range])

#Pee_ana = [Pee_analytical(L, E, theta12, theta13, Dmsq21, Dmsq31) for E in E_range] 