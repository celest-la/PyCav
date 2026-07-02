from __future__ import annotations
from typing import Union, Tuple, Optional

import torch
from pycav.physics.propagation import compute_steering_vectors

class BaseFrequencySolver:
    def __init__(self, probe: Probe, grid: Grid, c: float = 1540) -> None:
        self.probe  = probe
        self.grid = grid
        self.c = c
    @torch.no_grad()
    def get_steering_vectors(self, freqs: Union[torch.Tensor, List[float], float]) -> torch.Tensor:
        """Récupère A [F, P, M]"""
        return compute_steering_vectors(self.probe, self.grid.positions, freqs, self.c)

    def _format_output(self, img: torch.Tensor, average: bool) -> torch.Tensor:
        """
        Met en forme l'image finale pour correspondre à la taille de la grille (Nx, Nz).
        img a la forme [F, P] où P = Nx * Nz.
        """
        F = img.shape[0]
        # Reshape en [F, Nx, Nz]
        img_reshaped = img.view(F, *self.grid.shape)
        
        if average:
            return img_reshaped.mean(dim=0) # Renvoie [Nx, Nz]
        return img_reshaped # Renvoie [F, Nx, Nz]

class DAS(BaseFrequencySolver):
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], average: bool = True) -> torch.Tensor:
        a = self.get_steering_vectors(freqs)
        a= a/torch.tensor(self.probe.positions.shape[0])
        img = torch.einsum('fpi,fij,fpj->fp', a.conj(), csm, a).real
        return self._format_output(img, average)

class RCB(BaseFrequencySolver):
    @torch.no_grad()
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], epsilon: float = 0.1, average: bool = True)-> torch.Tensor:
        # a : [F, P, M]
        a = self.get_steering_vectors(freqs)
        
        # Inversion de la CSM avec diagonal loading
        identity = torch.eye(csm.shape[-1], device=csm.device, dtype=csm.dtype).unsqueeze(0)
        reg = epsilon * torch.mean(torch.diagonal(csm, dim1=-2, dim2=-1).real, dim=-1).view(-1, 1, 1)
        R = csm + reg * identity # R : [F, M, M]
        
        # Résolution du système linéaire R * x = a_transpose
        
        x = torch.linalg.solve(R, a.transpose(-1, -2))
        
        # Calcul du dénominateur : a^H * (R^-1 * a)
       
        den = torch.einsum('fpm,fmp->fp', a.conj(), x).real
        
        img = 1.0 / den
        return self._format_output(img, average)

class RCB_Li(BaseFrequencySolver):
    @torch.no_grad()
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], epsilon: float = 0.1, average: bool = True) -> torch.Tensor:
        a_all = self.get_steering_vectors(freqs)
        F, P, M = a_all.shape
        device = csm.device
        
        # Eigendecomposition
        # eigh garantit que L_val est STRICTEMENT RÉEL (float32)
        L_val, U = torch.linalg.eigh(csm)
        
        # SECURITÉ IMPORTANTE : on clamp les valeurs propres à 1e-12 
        # pour éviter les valeurs propres négatives dues aux imprécisions flottantes
        L_val = torch.clamp(L_val, min=1e-12)
        
        final_map = torch.zeros((F, P), device=device, dtype=torch.float32)

        # L'epsilon (tolérance) est une valeur purement RÉELLE
        eps_t = torch.tensor(epsilon, device=device, dtype=torch.float32)
        sqrt_eps = torch.sqrt(eps_t)

        for f in range(F):
            # Données locales
            a = a_all[f]
            L = L_val[f].unsqueeze(0)   # [1, M] (réel)
            U_f = U[f]                  # [M, M] (complexe)
            
            # 1. Projection
            sumU = a.conj() @ U_f               # [P, M] (complexe)
            
            # torch.abs() convertit nativement le complexe en float32 réel !
            abs_sumU2 = torch.abs(sumU)**2 # [P, M] (réel float32)
            
            # 2. Initialisation Lambda
            norm_a = torch.sqrt(torch.tensor(float(M), device=device, dtype=torch.float32)) 
            
            # lb est calculé comme un scalaire
            lb = (norm_a - sqrt_eps) / (L.max() * sqrt_eps)
            
            # On initialise lamb comme un VECTEUR de taille P (un par pixel)
            # max() gère le scalaire python, on remplit le tenseur
            lamb_init = 0.5 * max(lb.item(), 1e-10)
            lamb = torch.full((P,), lamb_init, device=device, dtype=torch.float32)

            # 3. Newton-Raphson Loop
            for _ in range(20):
                # denom: [P, M] car L est [1, M] et lamb.unsqueeze(1) est [P, 1]
                denom = 1 + L * lamb.unsqueeze(1) 
                
                f_val = torch.sum(abs_sumU2 / (denom**2), dim=1) - eps_t # [P]
                fp_val = -2 * torch.sum((L * abs_sumU2) / (denom**3), dim=1) # [P]
                
                lamb = lamb - f_val / fp_val
                lamb = lamb.clamp(min=1e-10) # Fonctionne parfaitement car lamb est réel

            # 4. Calcul Puissance
            term_pow = (lamb.unsqueeze(1) / (1 + lamb.unsqueeze(1) * L))**2
            denom_pow = torch.sum(L * abs_sumU2 * term_pow, dim=1)
            
            map_t = 1.0 / torch.clamp(denom_pow, min=1e-12)

            # 5. Correction d'Ambiguïté
            weighting = 1.0 / (1 + L * lamb.unsqueeze(1))
            norm_ac2 = torch.sum(abs_sumU2 * (1 - weighting)**2, dim=1)
            
            final_map[f] = map_t * (norm_ac2 / M)

        return self._format_output(final_map, average)
    
class FB(BaseFrequencySolver):
    @torch.no_grad()
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], r: float = 1, method: str = 'standard', average: bool = True)-> torch.Tensor:
        """
        Implémentation stricte du 'pisa' (Functional Beamforming) de ton MATLAB.
        
        Args:
            r (float): Paramètre de puissance.
                    r=1  -> DAS.
                    r=0  -> Midway.
                    r>0 -> Pisarenko
        """
        # 1. Récupération des Steering Vectors [F, P, M]
        a = self.get_steering_vectors(freqs)
        #a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True)
        #a=a/torch.tensor(self.probe.positions.shape[0])
        norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
        a = a / norm
        L, V = torch.linalg.eigh(csm)
        L = torch.clamp(L, min=1e-12)
        proj = torch.einsum('fpm,fmi->fpi', a.conj(), V)
        if r != 0:
            # On élève chaque valeur propre à la puissance 1/r

            L_pow = L**(1/r)
            
            # 3. Reconstruction de la matrice de poids W
            #W = torch.einsum('fmi,fi,fni->fmn', V, L_pow, V.conj())
            
            # 4. Calcul de la forme quadratique a^H * W * a
            #quad_form = torch.einsum('fpi,fij,fpj->fp', a.conj(), W, a).real
            quad_form = torch.einsum('fpi,fi,fpi->fp', proj, L_pow, proj.conj()).real
            # 5. Puissance finale element-wise sur l'image
            img= (quad_form**r)/self.probe.positions.shape[0]

        else:
            L_log = torch.log(L)
            W = torch.einsum('fmi,fi,fni->fmn', V, L_log, V.conj())
            log_form = torch.einsum('fpi,fij,fpj->fp', a.conj(), W, a).real
            img = torch.exp(log_form)/self.probe.positions.shape[0]


            

        return self._format_output(img, average)