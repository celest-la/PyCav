from __future__ import annotations
import torch
from typing import Union, List
from pycav.core.probe import Probe

@torch.no_grad()
def compute_steering_vectors(
    probe: Probe, 
    grid_positions: torch.Tensor, 
    freqs: Union[torch.Tensor, List[Union[float, torch.Tensor]], float], 
    c0: float = 1540.0
) -> torch.Tensor:
    
    # 1. Calcul de la matrice de distance [Pixels x Eléments]
    dist = torch.cdist(grid_positions, probe.positions, p=2)
    
    # 2. Calcul des délais [Pixels x Eléments]
    tau = dist / c0
    
    # 3. SÉCURITÉ : Nettoyage de `freqs`
    # Si l'utilisateur passe une liste (ex: [axf]), on extrait les valeurs en float purs
    if isinstance(freqs, list):
        freqs_clean = [f.item() if torch.is_tensor(f) else float(f) for f in freqs]
        f = torch.tensor(freqs_clean, dtype=torch.float32, device=dist.device)
    else:
        f = torch.as_tensor(freqs, dtype=torch.float32, device=dist.device)

    # 4. Passage au domaine fréquentiel [Fréquences x Pixels x Eléments]
    f = f.view(-1, 1, 1)
    
    # steer_vec = exp(-2j * pi * f * tau)
    # Déjà en complex64 car f et tau sont en float32 !
    A = torch.exp(-2j * torch.pi * f * tau.unsqueeze(0))
    
    return A

@torch.no_grad()
def compute_delay(
    probe: Probe, 
    grid_positions: torch.Tensor, 
    c0: float = 1540.0
) -> torch.Tensor:
    
    dist = torch.cdist(grid_positions, probe.positions, p=2)
    return dist / c0 * probe.fs