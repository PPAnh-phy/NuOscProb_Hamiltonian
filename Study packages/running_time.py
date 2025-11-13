from PMNS import *
from hamiltonian_vs_packages import *
import matplotlib.pyplot as plt
import time
# -----------------------------------
# User parameters
# ----------------------------------- 
N_eval = len(E_range)       
N_repeat = 50                 # number of repeated runs for statistics
methods = ["expm", "eigh", "NuFast", "NuExact"]

times_expm = []
times_eigh = []
times_nufast = []
times_nuexact = []

# -----------------------------------
# Benchmark loop
# -----------------------------------
for i in range(N_repeat):
    # Hamiltonian (expm)
    t0 = time.time()
    for E in E_range:
        P_3nu_evolutor_expm(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0)
    t1 = time.time()
    dt = t1 - t0
    times_expm.append(dt)

    # Hamiltonian (eigh)
    t0 = time.time()
    for E in E_range:
        P_3nu_evolutor_eigh(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0)
    t1 = time.time()
    dt = t1 - t0
    times_eigh.append(dt)
    
    # NuFast
    t0 = time.time()
    for E in E_range:
        Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_cp,
                               Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton)
    t1 = time.time()
    dt = t1 - t0
    times_nufast.append(dt)
   
    # NuExact
    t0 = time.time()
    for E in E_range:
        oscprob_nuexact_matter(L, E, Dmsq21, Dmsq31, U)
    t1 = time.time()
    dt = t1 - t0
    times_nuexact.append(dt)

# -----------------------------------
# Statistics
# -----------------------------------
def summarize(times):
    return np.mean(times), np.std(times), np.min(times), np.max(times)

mean_expm, std_expm, min_expm, max_expm = summarize(times_expm)
mean_eigh, std_eigh, min_eigh, max_eigh = summarize(times_eigh)
mean_nufast, std_nufast, min_nufast, max_nufast = summarize(times_nufast)
mean_nuexact, std_nuexact, min_nuexact, max_nuexact = summarize(times_nuexact)

print("\n===== Summary Statistics =====")
print(f"Hamiltonian (expm):  mean={mean_expm:.4f}s ±{std_expm:.4f}  [min={min_expm:.4f}, max={max_expm:.4f}]")
print(f"Hamiltonian (eigh):  mean={mean_eigh:.4f}s ±{std_eigh:.4f}  [min={min_eigh:.4f}, max={max_eigh:.4f}]")
print(f"NuFast:              mean={mean_nufast:.4f}s ±{std_nufast:.4f}  [min={min_nufast:.4f}, max={max_nufast:.4f}]")
print(f"NuExact:             mean={mean_nuexact:.4f}s ±{std_nuexact:.4f}  [min={min_nuexact:.4f}, max={max_nuexact:.4f}]")

# -----------------------------------
# Visualization
# -----------------------------------
means = [mean_expm, mean_eigh, mean_nufast, mean_nuexact]
stds = [std_expm, std_eigh, std_nufast, std_nuexact]

plt.figure(figsize=(8,5))
plt.bar(methods, means, yerr=stds, capsize=6, color=["#17becf", "#2ca02c", "#bcbd22", "#ff7f0e"])
plt.ylabel("Total runtime [s]")
plt.title(f"Average runtime for {N_eval} evaluations (over {N_repeat} runs)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Overall performance
N = 10000   # number of evaluations 
E0 = 0.002   # GeV, representative energy

print("\nThe running time of:")

# expm
t0 = time.time()
for _ in range(N):
    P_3nu_evolutor_expm(L, E0, U, [Dmsq21, Dmsq31], rho, 1, 0)
t1 = time.time()
print(f"Hamiltonian (expm): {t1 - t0:.4f} s for {N} evals")

# eigh
t0 = time.time()
for _ in range(N):
    P_3nu_evolutor_eigh(L, E0, U, [Dmsq21, Dmsq31], rho, 1, 0)
t1 = time.time()
print(f"Hamiltonian (eigh): {t1 - t0:.4f} s for {N} evals")

# NuFast
t0 = time.time()
for _ in range(N):
    Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_cp,
                           Dmsq21, Dmsq31, L, E0, rho, Ye, N_Newton)
t1 = time.time()
print(f"NuFast: {t1 - t0:.4f} s for {N} evals")

# NuExact
t0 = time.time()
for _ in range(N):
    oscprob_nuexact_matter(L, E0, Dmsq21, Dmsq31, U)
t1 = time.time()
print(f"NuExact: {t1 - t0:.4f} s for {N} evals")