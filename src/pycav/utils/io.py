import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt

import time

class Timer:
    """
    Accurate GPU/CPU timing context manager for PyTorch solvers.
    """
    def __init__(self, name="Solver", device="cpu"):
        self.name = name
        self.device = torch.device(device)
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        print(f"[{self.name}] Execution time: {self.elapsed_ms:.2f} ms ({self.elapsed_ms/1000.0:.3f} s)")

def load_verasonics_data(file_path):
    '''A faire : Charge les données Verasonics à partir d'un fichier .mat.'''


def get_rf_data(mat_data, buffer_idx=0):
    rf = mat_data['RcvData'][buffer_idx]
    return torch.from_numpy(rf.astype(np.float32))


def plot_rf_matrix(dirac_rf, fs, title="Données RF"):
    if torch.is_tensor(dirac_rf):
        dirac_rf = dirac_rf.detach().cpu().numpy()
        
    n_samples, nelem = dirac_rf.shape
    t_max = (n_samples / fs) * 1e6
    
    plt.figure(figsize=(8, 6))
    plt.imshow(dirac_rf, extent=[0, nelem, t_max, 0], aspect='auto', cmap='RdBu')
    plt.title(title)
    plt.ylabel("Temps (µs)")
    plt.xlabel("Index de l'élément")
    plt.colorbar(label="Amplitude")


def plot_rf_spectrum(rf_line, fs, f0=None, title="Spectre du signal RF"):
    if torch.is_tensor(rf_line):
        rf_line = rf_line.detach().cpu().numpy()
    
    n = len(rf_line)
    fft_values = np.fft.rfft(rf_line)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    
    magnitude = np.abs(fft_values)
    psd_db = 20 * np.log10(magnitude / (np.max(magnitude) + 1e-10))
    
    plt.figure(figsize=(10, 5))
    plt.plot(freqs / 1e6, psd_db, color='tab:blue', linewidth=1)
    
    if f0 is not None:
        f0_mhz = f0 / 1e6
        harmonics = [f0_mhz/2, f0_mhz, 1.5*f0_mhz, 2*f0_mhz, 2.5*f0_mhz, 3*f0_mhz]
        labels = ['f0/2', 'f0', '1.5f0', '2f0', '2.5f0', '3f0']
        
        for h, label in zip(harmonics, labels):
            plt.axvline(x=h, color='tab:red', linestyle='--', alpha=0.5)
            plt.text(h, plt.ylim()[1], label, color='tab:red', 
                     rotation=90, va='bottom', ha='center', fontsize=8)

    plt.title(title)
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, freqs.max() / 1e6 if freqs.max() < 10e6 else 10)
    plt.ylim(-100, 5)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()


def plot_result(img, grid, title):
    if img.ndim == 3:
        img = img.mean(dim=0)
    elif img.ndim == 1:
        img = img.view(*grid.shape)

    img_2d = img.detach().cpu().numpy()
    
    # Extract scalar extent values safely from PyTorch CUDA tensors
    x_min = float(grid.x.min().cpu().item() * 1e3)
    x_max = float(grid.x.max().cpu().item() * 1e3)
    z_min = float(grid.z.min().cpu().item() * 1e3)
    z_max = float(grid.z.max().cpu().item() * 1e3)

    plt.imshow(img_2d.T, extent=[x_min, x_max, z_max, z_min], cmap='hot', origin="upper")
    plt.colorbar(label='Intensité')
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")


def plot_result_dB(img, grid, title, dynamic_range=20):
    if img.ndim == 3:
        img = img.mean(dim=0)
    elif img.ndim == 1:
        img = img.view(*grid.shape)

    img_np = img.detach().cpu().numpy()
    
    img_rescale = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-12)
    img_safe = np.clip(img_rescale, 1e-12, None)
    img_db_norm = 10 * np.log10(img_safe / np.max(img_safe))
    
    # Extract scalar limits safely using .cpu().item()
    x_min = float(grid.x.min().cpu().item() * 1e3)
    x_max = float(grid.x.max().cpu().item() * 1e3)
    z_min = float(grid.z.min().cpu().item() * 1e3)
    z_max = float(grid.z.max().cpu().item() * 1e3)

    plt.imshow(
        img_db_norm.T, 
        extent=[x_min, x_max, z_max, z_min], 
        cmap='viridis', 
        vmin=-dynamic_range, 
        vmax=0, 
        origin="upper"
    )
               
    plt.colorbar(label='Intensité (dB)')
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")


def plot_result_3d_mip_dB(img, grid, title="Reconstruction 3D (MIP)", dynamic_range=20):
    """
    Affiche les projections d'intensité maximale (MIP) selon les 3 plans orthogonaux (XZ, YZ, XY).
    """
    if img.ndim == 4:  # [F, Nx, Ny, Nz]
        img = img.mean(dim=0)
        
    img_np = img.detach().cpu().numpy()  # [Nx, Ny, Nz]
    
    img_rescale = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-12)
    img_safe = np.clip(img_rescale, 1e-12, None)
    img_db = 10 * np.log10(img_safe / np.max(img_safe))
    
    # MIP Projections along X, Y, Z
    mip_xz = np.max(img_db, axis=1)  # Max along Y -> [Nx, Nz]
    mip_yz = np.max(img_db, axis=0)  # Max along X -> [Ny, Nz]
    mip_xy = np.max(img_db, axis=2)  # Max along Z -> [Nx, Ny]
    
    x_min, x_max = float(grid.x.min().cpu().item() * 1e3), float(grid.x.max().cpu().item() * 1e3)
    y_min, y_max = float(grid.y.min().cpu().item() * 1e3), float(grid.y.max().cpu().item() * 1e3)
    z_min, z_max = float(grid.z.min().cpu().item() * 1e3), float(grid.z.max().cpu().item() * 1e3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=14)
    
    # XZ Plan
    im0 = axes[0].imshow(mip_xz.T, extent=[x_min, x_max, z_max, z_min], cmap='viridis', vmin=-dynamic_range, vmax=0, origin="upper")
    axes[0].set_title("Plan X-Z (MIP Y)")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Z (mm)")
    
    # YZ Plan
    im1 = axes[1].imshow(mip_yz.T, extent=[y_min, y_max, z_max, z_min], cmap='viridis', vmin=-dynamic_range, vmax=0, origin="upper")
    axes[1].set_title("Plan Y-Z (MIP X)")
    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (mm)")
    
    # XY Plan
    im2 = axes[2].imshow(mip_xy.T, extent=[x_min, x_max, y_max, y_min], cmap='viridis', vmin=-dynamic_range, vmax=0, origin="upper")
    axes[2].set_title("Plan X-Y (MIP Z)")
    axes[2].set_xlabel("X (mm)")
    axes[2].set_ylabel("Y (mm)")
    
    fig.colorbar(im0, ax=axes.ravel().tolist(), label="Intensité (dB)")


def plot_bubble_positions_3d(positions,grid, probe=None, title="3D Geometry Setup (Probe & Bubbles)"):
    """
    Plots a 3D scatter view of simulated bubble positions and probe elements in mm.
    """
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot bubbles (X, Y, Z in mm)
    ax.scatter(
        positions[:, 0] * 1e3,
        positions[:, 1] * 1e3,
        positions[:, 2] * 1e3,
        c='red',
        marker='o',
        s=15,
        alpha=0.8,
        label=f'Bubbles (N={len(positions)})'
    )

    # Plot probe elements if provided
    if probe is not None:
        p_pos = probe.positions.detach().cpu().numpy()
        ax.scatter(
            p_pos[:, 0] * 1e3,
            p_pos[:, 1] * 1e3,
            p_pos[:, 2] * 1e3,
            c='blue',
            marker='s',
            s=12,
            alpha=0.5,
            label=f'Probe Elements (N={probe.n_elements})'
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(grid.x.min().cpu().item() * 1e3, grid.x.max().cpu().item() * 1e3)
    ax.set_ylim(grid.y.min().cpu().item() * 1e3, grid.y.max().cpu().item() * 1e3)
    ax.set_zlim(grid.z.min().cpu().item() * 1e3, grid.z.max().cpu().item() * 1e3)
    x_edges = np.array([grid.x.min().cpu().item() * 1e3, grid.x.max().cpu().item() * 1e3])
    y_edges = np.array([grid.y.min().cpu().item() * 1e3, grid.y.max().cpu().item() * 1e3])
    z_edges = np.array([grid.z.min().cpu().item() * 1e3, grid.z.max().cpu().item() * 1e3])
    x_span = abs(x_edges[-1] - x_edges[0])
    y_span = abs(y_edges[-1] - y_edges[0])
    z_span = abs(z_edges[-1] - z_edges[0])
    ax.set_box_aspect((x_span, y_span, z_span)) # FORCE TRUE 1:1:1 PHYSICAL ASPECT RATIO (voxels are rendered as true cubes)
    # Invert Z-axis so depth increases downwards (ultrasound convention)
    ax.invert_zaxis()

    plt.tight_layout()

def plot_volume_3d_voxels_dB(img, grid, title="Reconstructed 3D Volume", threshold_dB=-12.0, dynamic_range=20, show_full_grid_box=True):
    """
    Plots a 3D volumetric cloud of the beamformed acoustic map above threshold_dB.
    """

    if img.ndim == 4:  # [F, Nx, Ny, Nz]
        img = img.mean(dim=0)

    img_np = img.detach().cpu().numpy()  # [Nx, Ny, Nz]

    # Convert to dB
    img_rescale = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-12)
    img_safe = np.clip(img_rescale, 1e-12, None)
    img_db = 10 * np.log10(img_safe / np.max(img_safe))

    # Mask active voxels above threshold
    voxels = (img_db >= threshold_dB)

    # Get physical grid coordinates in mm
    x_mm = grid.x.cpu().numpy() * 1e3
    y_mm = grid.y.cpu().numpy() * 1e3
    z_mm = grid.z.cpu().numpy() * 1e3

    # Calculate bin edges for ax.voxels in physical mm
    dx = (x_mm[1] - x_mm[0]) if len(x_mm) > 1 else 1.0
    dy = (y_mm[1] - y_mm[0]) if len(y_mm) > 1 else 1.0
    dz = (z_mm[1] - z_mm[0]) if len(z_mm) > 1 else 1.0

    x_edges = np.append(x_mm - dx / 2.0, x_mm[-1] + dx / 2.0)
    y_edges = np.append(y_mm - dy / 2.0, y_mm[-1] + dy / 2.0)
    z_edges = np.append(z_mm - dz / 2.0, z_mm[-1] + dz / 2.0)

    # 3D Grid mesh for voxel edges
    X_e, Y_e, Z_e = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    # Map dB values to colors
    norm_db = (img_db - (-dynamic_range)) / dynamic_range
    norm_db = np.clip(norm_db, 0, 1)

    colors = plt.cm.viridis(norm_db)
    colors[..., 3] = np.where(voxels, 0.65, 0.0)  # Alpha transparency for active voxels

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    if np.any(voxels):
        ax.voxels(X_e, Y_e, Z_e, voxels, facecolors=colors, edgecolors=None)
    else:
        print(f"[Warning] No voxels above threshold {threshold_dB} dB.")

    # Draw full original grid bounding box wireframe
    if show_full_grid_box:
        x0, x1 = x_edges[0], x_edges[-1]
        y0, y1 = y_edges[0], y_edges[-1]
        z0, z1 = z_edges[0], z_edges[-1]
        
        corners = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e in edges:
            p1, p2 = corners[e[0]], corners[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', linestyle='--', alpha=0.5)

    # Set full physical grid limits
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_zlim(z_edges[-1], z_edges[0])  # Invert Z so depth increases downwards

    # FORCE TRUE 1:1:1 PHYSICAL ASPECT RATIO (voxels are rendered as true cubes)
    x_span = abs(x_edges[-1] - x_edges[0])
    y_span = abs(y_edges[-1] - y_edges[0])
    z_span = abs(z_edges[-1] - z_edges[0])
    ax.set_box_aspect((x_span, y_span, z_span))

    ax.set_title(f"{title} (Isosurface >= {threshold_dB} dB)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    plt.tight_layout()