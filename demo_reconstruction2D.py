import torch
import numpy as np
import matplotlib.pyplot as plt
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.solvers.beamformers import DAS, RCB, FB
from pycav.simulation.simulation import BBagSimulator, getPositions, StableSimulator
from pycav.utils.io import plot_result_dB, Timer
from pycav.solvers.preproc import compute_csm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)

# 1. SETUP GÉOMÉTRIQUE & MATÉRIEL
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

probe = Probe.from_linear(n_el=128, pitch=0.3e-3, device=device)
grid = Grid.from_limits(
    x_lim=(-0.01, 0.01), 
    z_lim=(0.06, 0.08), 
    step=0.2e-3, 
    device=device)

positions, center, x_size, z_size, angle = getPositions.random_ellipse(
    x_position_lim=-0.004, 
    z_position_lim=0.07, 
    x_size_lim=2e-3, 
    z_size_lim=2e-3, 
    angle=0, 
    density=100,
    device=device  # <-- Now created directly on CUDA
)

fhifu = 1e6
c0 = 1480

# 2. SIMULATION SIGNAL RF & CSM
'''sim_bbag = BBagSimulator(probe, positions, fhifu=fhifu, device=device, c0=c0)
rf_filtered = sim_bbag.simulate(t_acq=5e-4, n_cycles=200, noise_level=0.001)
'''

sim_stable = StableSimulator(probe, positions, fhifu=fhifu, device=device, c0=c0)
rf_filtered = sim_stable.simulate(t_acq=5e-4, n_cycles=200, noise_level=0.001)
# 3.CLEAR PYTORCH CUDA CACHE after simulation to free up memory for reconstruction
torch.cuda.empty_cache()
del sim_stable
csm, axf = compute_csm(rf_filtered, probe, K=130, overlap=0.9, f=[5.5e6])
freqs = axf


# 4. SOLVERS RECONSTRUCTION
das = DAS(probe, grid, c=c0)
rcb = RCB(probe, grid, c=c0)
fb = FB(probe, grid, c=c0)


with Timer("DAS", device=device):
    img_das = das.solve(csm, freqs, average=True)
with Timer("RCB", device=device):
    img_rcb = rcb.solve(csm, freqs, epsilon=5, average=True)
with Timer("FB", device=device):
    img_fb = fb.solve(csm, freqs, r=0, average=True)


# 5. AFFICHAGE COMPARATIF
plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plot_result_dB(img_das, grid, "DAS", dynamic_range=20)

plt.subplot(1, 3, 2)
plot_result_dB(img_rcb, grid, "RCB", dynamic_range=20)

plt.subplot(1, 3, 3)
plot_result_dB(img_fb, grid, "FB (PISA)", dynamic_range=20)

plt.tight_layout()
plt.savefig("comparatif_beamforming.png", dpi=300, bbox_inches='tight')
print("Succès : L'image a été sauvegardée sous le nom 'comparatif_beamforming.png'")