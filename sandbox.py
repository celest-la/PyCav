# %%
import torch
import matplotlib.pyplot as plt
from pycav.core.probe import Probe
from pycav.core.grid import Grid
#from pycav.solvers.beamformers import DAS, RCB, FB,RCB_Li


# %%
# 1. SETUP GÉOMÉTRIQUE
device = "cuda" if torch.cuda.is_available() else "cpu"

# Création d'une sonde linéaire (128 élem, pitch 0.3mm)
probe = Probe.from_linear(n_el=128, pitch=0.3e-3, device=device)

# Création d'une grille de recherche (2cm x 2cm à partir de 1cm de profondeur)
grid = Grid.from_limits(x_lim=(-0.01, 0.01), z_lim=(0.01, 0.03), step=0.0002, device=device)

fhifu = 1.2e6  # 2 MHz
c0 = 1540
bubble_positions = torch.tensor([[0.002, 0.0, 0.02]
                          ,[0.003, 0.0, 0.022]], device=device) # Source décalée à x=2mm, z=20mm

# %%
A0 = 1
n_cycles=100
t0=0
t_acq = 200e-6 #temps pendant lequel la sonde écoute
nb = bubble_positions.shape[0]
nelem = probe.positions.shape[0]

t_delay=torch.cdist(probe.positions,bubble_positions,p=2)
index_tau=t_delay/c0*probe.fs+t0

# Create the dirac
amplitudes = A0 + 0.5*torch.rand(n_cycles,nb)
index_dirac = (torch.arange(n_cycles, device=device).float()+0.5) * probe.fs/fhifu

# %%
t_temp = index_tau.unsqueeze(-1) + index_dirac.view(1, 1, -1)


tau_final = torch.permute(t_temp, (0,2,1))
# %%
n_samples = int(t_acq*probe.fs)
dirac_RF = torch.zeros(n_samples,nelem)

tau_floor=torch.floor(tau_final)
frac=tau_final-tau_floor

# %%

# Préparation des valeurs pondérées par l'amplitude de la bulle
# amplitudes (Nb,) est broadcasté sur (Ne, Nb)
val_floor = (amplitudes * (1 - frac))
val_ceil = (amplitudes * frac)

# %% 1. Préparation des dimensions
# tau_final a la forme (nelem, n_cycles, nbulles) après ton permute
nelem, n_cycles, nbulles = tau_final.shape
device = tau_final.device

# On crée une grille d'indices pour les éléments (nelem) 
# qu'on répète pour matcher la structure de tau_final
# shape: (nelem, n_cycles, nbulles)
elem_indices = torch.arange(nelem, device=device).view(-1, 1, 1).expand(-1, n_cycles, nbulles)

# %% 2. Aplatissement pour le Splatting global
# On transforme tout en vecteurs 1D pour injecter d'un coup dans la matrice RF
t_idx_f = tau_floor.reshape(-1).long()
elem_idx = elem_indices.reshape(-1)
vals_f = val_floor.reshape(-1)

t_idx_c = (tau_floor + 1).reshape(-1).long()
vals_c = val_ceil.reshape(-1)

# %% 3. Application des masques
mask_f = (t_idx_f >= 0) & (t_idx_f < n_samples)
mask_c = (t_idx_c >= 0) & (t_idx_c < n_samples)

# %% 4. Splatting final
# Attention : dirac_RF est (n_samples, nelem)
# Donc l'indexation est (temps, element)
dirac_RF.index_put_((t_idx_f[mask_f], elem_idx[mask_f]), vals_f[mask_f], accumulate=True)
dirac_RF.index_put_((t_idx_c[mask_c], elem_idx[mask_c]), vals_c[mask_c], accumulate=True)