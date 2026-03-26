# %%
import torch
import matplotlib.pyplot as plt
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.solvers.beamformers import DAS, RCB, FB,RCB_Li

# %%
# 1. SETUP GÉOMÉTRIQUE
device = "cuda" if torch.cuda.is_available() else "cpu"

# Création d'une sonde linéaire (128 élem, pitch 0.3mm)
probe = Probe.from_linear(n_el=128, pitch=0.3e-3, device=device)

# Création d'une grille de recherche (2cm x 2cm à partir de 1cm de profondeur)
grid = Grid.from_limits(x_lim=(-0.01, 0.01), z_lim=(0.01, 0.03), step=0.0002, device=device)

# 2. SIMULATION D'UNE SOURCE (La "Vérité")
f_source = 1e6  # 2 MHz
c0 = 1540
source_pos = torch.tensor([0.002, 0.0, 0.02], device=device) # Source décalée à x=2mm, z=20mm

# Calcul des délais théoriques entre la source et chaque élément de la sonde
dist_source = torch.linalg.vector_norm(probe.positions - source_pos,dim=-1)
# Signal complexe reçu sur chaque capteur (vecteur de phase)
X = 1*torch.exp(-1j * 2 * torch.pi * f_source * dist_source / c0)

# Génération de la CSM théorique (Produit extérieur)
# On rajoute une dimension pour simuler le batch de fréquences [F=1, M, M]
csm = torch.outer(X, X.conj()).unsqueeze(0)

# 3. RECONSTRUCTION
# Initialisation des solvers
das = DAS(probe, grid, c=c0)
rcb = RCB(probe, grid, c=c0)
fb = FB(probe, grid, c=c0)
# Exécution
freqs = [f_source]
# %%
img_das = das.solve(csm, freqs)
img_rcb = rcb.solve(csm, freqs, epsilon=10) # Très peu de bruit donc epsilon faible
img_fb = fb.solve(csm, freqs, r=0) 

# %%
#rcb_li = RCB_Li(probe, grid, c=c0)
#img_rcb = rcb_li.solve(csm, freqs, epsilon=10)

# %%
# 4. VISUALISATION
def plot_result(img, title):
    # On reshape l'image pour l'affichage (Grid stocke le shape)
    img_2d = img.reshape(grid.shape).cpu().numpy()
    
    plt.imshow(img_2d.T, extent=[-10, 10, 30, 10], cmap='hot')
    plt.colorbar(label='Intensité')
    plt.scatter(source_pos[0].cpu()*1e3, source_pos[2].cpu()*1e3, marker='x', color='cyan', label='Vraie Source')
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.legend()

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1); plot_result(img_das, "DAS - Point Source")
plt.subplot(1, 3, 2); plot_result(img_rcb, "RCB - Point Source")
plt.subplot(1, 3, 3); plot_result(img_fb, "FB - Point Source")
plt.tight_layout()
plt.show()
# %%
