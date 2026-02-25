# %%
import matplotlib.pyplot as plt
from pycav.core import Probe

# Création de la probe
probe = Probe.from_linear(n_el=128, fs=20e6, pitch=0.3e-3)

# Visualisation
plt.figure(figsize=(10, 2))
plt.scatter(probe.positions[:, 0]*1e3, probe.positions[:, 2]*1e3, marker='|')
plt.title(f"Sonde {probe.n_elements} éléments")
plt.xlabel("X (mm)")
plt.ylabel("Z (mm")
plt.axis('equal')
plt.show()

print(f"Sonde créée sur : {probe.device}")
# %%
