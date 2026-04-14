import torch

def compute_steering_vectors(probe, grid_positions, freqs, c0=1540):
    
    # 1. Calcul de la matrice de distance [Pixels x Eléments]
    # Remplace sqrt((X-xp)^2 + (Z-zp)^2)
    dist = torch.cdist(grid_positions, probe.positions, p=2)
    
    # 2. Calcul des délais [Pixels x Eléments]
    tau = dist / c0
    
    # 3. Passage au domaine fréquentiel [Fréquences x Pixels x Eléments]
    # On ajoute une dimension aux fréquences pour le broadcasting
    f = torch.as_tensor(freqs, device=dist.device).view(-1, 1, 1)
    
    # steer_vec = exp(-2j * pi * f * tau)
    # Le .unsqueeze(0) permet d'aligner tau [1, P, M] avec f [F, 1, 1]
    A = torch.exp(-2j * torch.pi * f * tau.unsqueeze(0))
    
    return A.to(torch.complex64)

def compute_delay(probe, grid_positions, c0=1540):
    # 1. Calcul de la matrice de distance [Pixels x Eléments]
    # grid_position peut être une grille ou des bulles (cas de la simulation)
    dist = torch.cdist(grid_positions, probe.positions, p=2)

    # 2. Calcul des indices (brut, sans interpolation)
    return dist / c0 * probe.fs