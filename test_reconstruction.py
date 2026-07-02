import torch
import matplotlib.pyplot as plt
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.solvers.beamformers import DAS, RCB, FB,RCB_Li
from pycav.simulation.simulation import StableSimulator, BBagSimulator, getPositions, Stable
from pycav.utils.io import plot_rf_matrix, plot_rf_spectrum, plot_result_dB, plot_result
import numpy as np
from pycav.solvers.preproc import compute_csm

torch.set_default_dtype(torch.float32) # <- Force PyTorch à utiliser le simple précision par défaut
# 1. SETUP GÉOMÉTRIQUE
device = "cuda" if torch.cuda.is_available() else "cpu"

# Création d'une sonde linéaire (128 élem, pitch 0.3mm)
probe = Probe.from_linear(n_el=128, pitch=0.3e-3, device=device)

# Création d'une grille de recherche (2cm x 2cm à partir de 1cm de profondeur)
grid = Grid.from_limits(x_lim=(-0.01, 0.01), z_lim=(0.06, 0.08), step=0.0002, device=device)

# Source ponctuelle à x=2mm, z=20mm

positions, center, x_size, z_size, angle = getPositions.random_ellipse(x_position_lim=(-0.002, 0.002), z_position_lim=0.07, x_size_lim=2e-3, z_size_lim=2e-3, angle=0, density=100)

fhifu=1e6
c0=1480

'''sim_bbag = BBagSimulator(probe, positions, fhifu=fhifu, device=device, c0=c0)

rf_filtered = sim_bbag.simulate(t_acq=5e-4, n_cycles=200, noise_level=0.001)'''


sim_stable = StableSimulator(probe, positions, fhifu=fhifu, device=device, c0=c0)

# UNE SEULE LIGNE ! (Avec contrôle total de la signature de la bulle)
rf_filtered = sim_stable.simulate(t_acq=1e-3, n_cycles=500, noise_level=0.001)


csm, axf = compute_csm(rf_filtered, probe, K=130, overlap=0.9, f=[4.5e6])

# 3. RECONSTRUCTION
# Initialisation des solvers
das = DAS(probe, grid, c=c0)
rcb = RCB_Li(probe, grid, c=c0)
fb = FB(probe, grid, c=c0)
# Exécution
freqs = [axf[0]]  # On prend la première fréquence pour la reconstruction
# %%
img_das = das.solve(csm, freqs)
img_rcb = rcb.solve(csm, freqs, epsilon=5)
img_fb = fb.solve(csm, freqs, r=0) 


plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1) 
plot_result_dB(img_das, grid, "DAS - Point Source", dynamic_range=20)
plt.subplot(1, 3, 2); plot_result_dB(img_rcb, grid, "RCB - Point Source", dynamic_range=20)
plt.subplot(1, 3, 3); plot_result_dB(img_fb, grid, "FB - Point Source", dynamic_range=20)
plt.tight_layout()
plt.show()
