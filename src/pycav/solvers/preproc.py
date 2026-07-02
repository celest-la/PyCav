from __future__ import annotations
from typing import Union, Tuple, Optional

import torch
import numpy as np


@torch.no_grad()
def compute_csm(raw: torch.Tensor, probe: Probe, K: int, overlap: float, f: Union[torch.Tensor, List[float], float], n_bin: int = 1, n_zp: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule la Cross-Spectral Matrix (CSM) par la méthode des snapshots.
    
    Args:
        raw: Tenseur [Nt, Ne] (Temps, Éléments)
        probe.fs: Fréquence d'échantillonnage
        K: Nombre de snapshots souhaités
        overlap: Pourcentage de recouvrement (0.0 à 1.0)
        f: Liste ou tenseur des fréquences de reconstruction
        n_bin: Nombre de bins fréquentiels autour de f
        n_zp: Nombre de points pour la FFT (Zero-padding)
        
    Returns:
        CSM: [F, Ne, Ne] (Moyennée sur les snapshots)
        axf: Fréquences réelles correspondant aux bins sélectionnés
    """
    if raw.dtype != torch.float32:
        raw = raw.to(torch.float32)

    fs=probe.fs
    device = raw.device
    Nt, Ne = raw.shape
    f = torch.as_tensor(f, device=device, dtype=torch.float32)

    # 1. Définition des variables de fenêtrage (Logique MATLAB)
    nw = int(Nt // (K * (1 - overlap) + overlap))
    lag = int(round((1 - overlap) * nw))
    
    # Calcul des indices de départ des snapshots
    starts = torch.arange(0, Nt - nw + 1, lag, device=device)
    n_snapshots = len(starts)

    # 2. Gestion du Zero-Padding (Nzp)
    if n_zp == 0:
        delta_f = 0.01 * f.min()
        n_zp = int(round(fs / delta_f.item()))
        if n_zp < nw:
            n_zp = nw
            
    # On arrondit à la puissance de 2 supérieure pour la vitesse FFT
    n_zp = int(2**np.ceil(np.log2(n_zp)))

    # 3. Identification des bins fréquentiels
    vec_freq = torch.arange(n_zp, device=device,dtype=torch.float32) * (fs / n_zp)
    
    # Trouver les indices les plus proches des fréquences f
    # On utilise broadcasting pour comparer toutes les fréquences d'un coup
    diff = torch.abs(vec_freq.view(1, -1) - f.view(-1, 1))
    idx_center = torch.argmin(diff, dim=1)
    
    # Gestion du multi-bin (n_bin)
    if n_bin > 1:
        offsets = torch.arange(-(n_bin // 2), n_bin // 2 + (n_bin % 2), device=device)
        indices = idx_center.view(-1, 1) + offsets.view(1, -1)
    else:
        indices = idx_center.view(-1, 1)

    # Fréquences réelles des bins
    axf = vec_freq[indices]

    # 4. Préparation de la fenêtre (Tukey/Hann)
    # On utilise une fenêtre de Hann (proche du Tukey MATLAB par défaut)
    win = torch.hann_window(nw, periodic=False, device=device, dtype=torch.float32).view(-1, 1)

    # 5. Calcul des FFT par snapshots
    # On va stocker les spectres complexes : [n_snapshots, n_freq_indices, Ne]
    #tf_f = torch.zeros((n_snapshots, indices.numel(), Ne), dtype=torch.complex64, device=device)
    '''
    for i, start in enumerate(starts):
        # Extraction du snapshot + application de la fenêtre
        snapshot = raw[start : start + nw, :] * win
        
        # FFT sur n_zp points
        full_fft = torch.fft.fft(snapshot, n=n_zp, dim=0)
        
        # On ne garde que les bins qui nous intéressent
        # indices.flatten() contient tous les bins pour toutes les fréquences demandées
        tf_f[i] = full_fft[indices.flatten(), :]'''
    # raw shape: [Nt, Ne]
    snapshots = raw.unfold(dimension=0, size=nw, step=lag) # [n_snapshots, Ne, nw]
    snapshots = snapshots.permute(0, 2, 1) # [n_snapshots, nw, Ne]

    # Application de la fenêtre (broadcast)
    snapshots = snapshots * win.view(1, -1, 1)

    # FFT sur tout le batch d'un coup ! 🔥
    full_fft = torch.fft.fft(snapshots, n=n_zp, dim=1)

    # Extraction des bins
    tf_f = full_fft[:, indices.flatten(), :] # [n_snapshots, n_bins, Ne]

    # 6. Calcul de la CSM : Moyenne de (X * X^H)
    # X est [n_snapshots, n_bins, Ne]
    # On veut CSM : [n_bins, Ne, Ne]
    
    # On réorganise pour le produit extérieur batché
    # [n_bins, n_snapshots, Ne]
    S = tf_f.transpose(0, 1)
    
    # Produit extérieur batché : (bin, snap, Ne, 1) x (bin, snap, 1, Ne)
    # On moyenne sur la dimension 'snap' (n_snapshots)
    csm_per_bin = torch.einsum('bse,bsm->bem', S, S.conj()) / n_snapshots
    
    # Si nBin > 1, on regroupe et on moyenne par fréquence cible
    if n_bin > 1:
        # On redimensionne : [F, nBin, Ne, Ne]
        csm_reshaped = csm_per_bin.view(len(f), n_bin, Ne, Ne)
        # On moyenne sur les bins pour chaque fréquence : [F, Ne, Ne]
        csm_final = csm_reshaped.mean(dim=1)
        # On prend la fréquence centrale pour le steering vector
        axf_final = axf[:, n_bin // 2] 
    else:
        csm_final = csm_per_bin
        axf_final = axf.flatten()

    return csm_final, axf_final