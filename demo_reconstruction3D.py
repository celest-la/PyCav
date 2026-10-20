import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.solvers.beamformers import DAS, RCB, FB
from pycav.simulation.simulation import BBagSimulator, getPositions
from pycav.solvers.preproc import compute_csm
from pycav.utils.io import plot_result_3d_mip_dB, plot_bubble_positions_3d, plot_volume_3d_voxels_dB, Timer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)

# 1. HARDWARE & DEVICE SETUP
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

# 2. 3D PROBE & 3D GRID GEOMETRY
# Create a 2D Matrix Array Probe (16 x 16 = 256 elements, pitch 0.3mm)
mat_file = "src/pycav/utils/matrix1024.mat"

probe = Probe.from_matlab_file(mat_file, struct_name='param', device=device)

# Create a 3D Imaging Grid [Nx x Ny x Nz]
grid = Grid.from_limits_3d(
    x_lim=(-5e-3, 5e-3),  # 1 cm width
    y_lim=(-5e-3, 5e-3),  # 1 cm height
    z_lim=(10e-3, 30e-3),   # 1 cm depth
    step=0.2e-3,            # 0.2 mm step
    device=device
)

print(f"3D Probe Elements : {probe.n_elements}")
print(f"3D Grid Voxels     : {grid.n_pixels} (Shape: {grid.shape})")

# 3. 3D BUBBLE CLOUD SIMULATION
# Generate 3D bubble positions within an ellipsoid centered at z=20mm
positions, center, axes = getPositions.random_ellipsoid(
    x_position_lim=0.5e-3,
    y_position_lim=1.0e-3,
    z_position_lim=22e-3,
    x_size_lim=2.0e-3,
    y_size_lim=1.0e-3,
    z_size_lim=3.0e-3,
    density=1000,
    device=device
)

c0 = 1480.0
fhifu = 1.0e6

sim = BBagSimulator(probe, positions, fhifu=fhifu, device=device, c0=c0)
rf_data = sim.simulate(t_acq=4e-4, n_cycles=150, noise_level=0.001, batch_size=500)

# CLEAR PYTORCH CUDA CACHE after simulation to free up memory for reconstruction
torch.cuda.empty_cache()
del sim

plot_bubble_positions_3d(positions, grid, title="3D Bubble Cloud Simulation")
# Compute Cross-Spectral Matrix (CSM)
csm, freqs = compute_csm(rf_data, probe, K=80, overlap=0.8, f=[3.5e6])

# 4. 3D RECONSTRUCTION WITH SOLVERS
das = DAS(probe, grid, c=c0)
rcb = RCB(probe, grid, c=c0)
fb = FB(probe, grid, c=c0)


print("\n--- Running 3D Solvers ---")
torch.cuda.empty_cache()
print("1. DAS 3D...")
with Timer("DAS 3D", device=device):
    img_das = das.solve(csm, freqs, average=True)
torch.cuda.empty_cache()
print("2. RCB 3D...")
with Timer("RCB 3D", device=device):
    img_rcb = rcb.solve(csm, freqs, epsilon=5, average=True)
torch.cuda.empty_cache()
print("3. FB (PISA) 3D...")
with Timer("FB 3D", device=device):
    img_fb = fb.solve(csm, freqs, r=0, average=True)

torch.cuda.empty_cache()


# 5. VISUALIZATION (Maximum Intensity Projections - MIP for XZ, YZ, XY planes)
plot_result_3d_mip_dB(img_das, grid, title="3D DAS Reconstruction (MIP)", dynamic_range=10)
plot_result_3d_mip_dB(img_rcb, grid, title="3D RCB Reconstruction (MIP)", dynamic_range=10)
plot_result_3d_mip_dB(img_fb, grid, title="3D FB (PISA) Reconstruction (MIP)", dynamic_range=10)

#plot_volume_3d_voxels_dB(img_cmf, grid, title="CMF-spTV 3D Bubble Cloud", threshold_dB=-12.0)
#plot_volume_3d_voxels_dB(img_cmfa, grid, title="CMF-spTV Adaptive 3D Bubble Cloud", threshold_dB=-12.0)
#plot_volume_3d_voxels_dB(img_fb, grid, title="FB 3D Bubble Cloud", threshold_dB=-12.0)
#plot_volume_3d_voxels_dB(img_rcb, grid, title="RCB 3D Bubble Cloud", threshold_dB=-12.0)
plt.show()