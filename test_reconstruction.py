import torch
import matplotlib.pyplot as plt
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.solvers.beamformers import DAS, RCB, FB,RCB_Li
from pycav.simulation.simulation import StableSimulator, BBagSimulator, getPositions, Stable
from pycav.utils.io import plot_rf_matrix, plot_rf_spectrum, plot_result_dB, plot_result
import numpy as np
from pycav.solvers.preproc import compute_csm

# 1. SETUP GÉOMÉTRIQUE
device = "cuda" if torch.cuda.is_available() else "cpu"

# Création d'une sonde linéaire (128 élem, pitch 0.3mm)
probe = Probe.from_linear(n_el=128, pitch=0.3e-3, device=device)

# Création d'une grille de recherche (2cm x 2cm à partir de 1cm de profondeur)
grid = Grid.from_limits(x_lim=(-0.01, 0.01), z_lim=(0.06, 0.08), step=0.0002, device=device)

# Source ponctuelle à x=2mm, z=20mm

positions, center, x_size, z_size, angle = getPositions.random_ellipse(x_position_lim=(-0.002, 0.002), z_position_lim=0.065, x_size_lim=2e-3, z_size_lim=2e-3, angle=0, density=100)

#positions = torch.tensor([[0.002, 0.0, 0.02]], device=device)
positions_numpy= positions.cpu().numpy()
plt.figure()
plt.scatter(positions_numpy[:,0]*1e3, positions_numpy[:,2]*1e3)

plt.xlabel("X (mm)")
plt.ylabel("Z (mm)")    
plt.axis('equal')
plt.xlim(grid.x.min()*1e3, grid.x.max()*1e3)
plt.ylim(grid.z.max()*1e3, grid.z.min()*1e3)

fhifu=1e6
c0=1480

sim = StableSimulator(probe, positions, fhifu=fhifu, device=device,c0=c0)
diracs = sim.get_delayed_dirac(n_cycles=500, t_acq=1e-3)
plt.figure()
plt.plot(diracs.cpu().numpy()[1000:1400, 64])
k_sub = Stable(probe.fs, fhifu).get_kernel(alpha=1.0, beta=0.5, delta=0.2)


# --- 3. Convolution ---
rf_final = sim.apply_kernel(diracs, k_sub)




sim=BBagSimulator(probe, positions, fhifu=fhifu, device=device,c0=c0)
rf_final=sim.get_RF(t_acq=5e-4)

# 1. Générer un bruit blanc sur toute la matrice RF
# On calibre l'amplitude par rapport au max du signal (ex: 5%)
noise = 0.001 * torch.randn_like(rf_final) * rf_final.max()

# 2. Optionnel : Filtrer le bruit pour qu'il soit "coloré" par la sonde
# (ou simplement l'ajouter à la fin si tu considères que c'est du bruit d'acquisition)
rf_final= rf_final + noise


rf_filtered = sim.apply_probe_filter(rf_final)


'''# --- 4. Visualisation ---
plot_rf_matrix(rf_filtered, probe.fs) # La fonction qu'on a faite avant


# Transfert sur CPU pour Matplotlib
k_cpu = k_sub.cpu().numpy()
n_samples1 = k_cpu.shape[0]
print(k_cpu[0])
print(k_cpu[-1])
time_axis = np.arange(n_samples1) / probe.fs * 1e6
plt.plot(time_axis, k_cpu, label='Kernel Stable (2 Cycles)', color='blue', linewidth=1.5)

# Exemple sur l'élément central
line_idx = rf_final.shape[1] // 2
plot_rf_spectrum(rf_filtered[:, line_idx], fs=probe.fs, f0=1e6)

channel = 64
rf_line = rf_filtered[1000:1400, channel].cpu().numpy() # .cpu() si tu es sur GPU
n_samples = rf_filtered.shape[0]
time_axis = np.arange(n_samples) / probe.fs * 1e6
plt.figure(figsize=(10, 4))
plt.plot(time_axis[1000:1400], rf_line)
plt.title(f"Signal RF - Élément {channel}")
plt.xlabel("microsecondes (µs)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()'''

csm, axf = compute_csm(rf_filtered, probe, K=130, overlap=0.9, f=[4.5e6])
'''phase_csm = torch.angle(torch.squeeze(csm[0, :, :]))

plt.figure()
plt.imshow(phase_csm.cpu().numpy(), cmap='twilight')
plt.axis('equal')
plt.colorbar(label='Phase (radians)')
plt.show()'''
'''
# Génération de la CSM théorique (Produit extérieur)
# On rajoute une dimension pour simuler le batch de fréquences [F=1, M, M]
csm = torch.outer(X, X.conj()).unsqueeze(0)
'''
# 3. RECONSTRUCTION
# Initialisation des solvers
das = DAS(probe, grid, c=c0)
rcb = RCB(probe, grid, c=c0)
fb = FB(probe, grid, c=c0)
# Exécution
freqs = [axf[0]]  # On prend la première fréquence pour la reconstruction
# %%
img_das = das.solve(csm, freqs)
img_rcb = rcb.solve(csm, freqs, epsilon=10) # Très peu de bruit donc epsilon faible
img_fb = fb.solve(csm, freqs, r=0) 

# %%
#rcb_li = RCB_Li(probe, grid, c=c0)
#img_rcb = rcb_li.solve(csm, freqs, epsilon=10)

# %%
# 4. VISUALISATION


plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1) 
plot_result_dB(img_das, grid, "DAS - Point Source", dynamic_range=20)
plt.subplot(1, 3, 2); plot_result_dB(img_rcb, grid, "RCB - Point Source", dynamic_range=20)
plt.subplot(1, 3, 3); plot_result_dB(img_fb, grid, "FB - Point Source", dynamic_range=20)
plt.tight_layout()
plt.show()
