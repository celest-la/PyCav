import torch

def compute_steering_vectors(probe, grid, freqs, c0=1540):
    """
    Traduction directe de ta logique MATLAB en PyTorch optimisé.
    """
    # 1. Calcul de la matrice de distance [Pixels x Eléments]
    # Remplace sqrt((X-xp)^2 + (Z-zp)^2)
    dist = torch.cdist(grid.positions, probe.positions, p=2)
    
    # 2. Calcul des délais [Pixels x Eléments]
    tau = dist / c0
    
    # 3. Passage au domaine fréquentiel [Fréquences x Pixels x Eléments]
    # On ajoute une dimension aux fréquences pour le broadcasting
    f = torch.as_tensor(freqs, device=dist.device).view(-1, 1, 1)
    
    # steer_vec = exp(-2j * pi * f * tau)
    # Le .unsqueeze(0) permet d'aligner tau [1, P, M] avec f [F, 1, 1]
    A = torch.exp(-2j * torch.pi * f * tau.unsqueeze(0))
    
    return A.to(torch.complex64)