import torch
from pycav.physics.propagation import compute_steering_vectors

class BaseFrequencySolver:
    def __init__(self, probe, grid, c=1540):
        self.probe = probe
        self.grid = grid
        self.c = c

    def get_steering_vectors(self, freqs):
        """Récupère A [F, P, M]"""
        return compute_steering_vectors(self.probe, self.grid, freqs, self.c)

class DAS(BaseFrequencySolver):
    def solve(self, csm, freqs, average=True):
        """Prend la CSM déjà calculée en entrée"""
        a = self.get_steering_vectors(freqs)
        #vec_a = torch.linalg.vector_norm(a)
        a = a / torch.linalg.vector_norm(a)
        # Image = diag(A^H * CSM * A)
        img = torch.einsum('fpi,fij,fpj->fp', a.conj(), csm, a).real
        return img.mean(dim=0) if average else img

class RCB(BaseFrequencySolver):
    def solve(self, csm, freqs, epsilon=0.1, average=True):
        a = self.get_steering_vectors(freqs)
        
        # Inversion de la CSM avec diagonal loading
        # On traite tout le batch de fréquences d'un coup
        identity = torch.eye(csm.shape[-1], device=csm.device).unsqueeze(0)
        reg = epsilon * torch.mean(torch.diagonal(csm, dim1=-2, dim2=-1).real, dim=-1).view(-1, 1, 1)
        csm_inv = torch.inverse(csm + reg * identity)
        
        den = torch.einsum('fpi,fij,fpj->fp', a.conj(), csm_inv, a).real
        img = 1.0 / den
        return img.mean(dim=0) if average else img

class RCB_Li(BaseFrequencySolver):
    def solve(self, csm, freqs, epsilon=0.1, average=True):
        """
        Version corrigée : 100% PyTorch, zéro NumPy.
        """
        a_all = self.get_steering_vectors(freqs)
        F, P, M = a_all.shape
        device = csm.device  # On récupère le device (CPU ou CUDA)
        
        # Eigendecomposition
        L_val, U = torch.linalg.eigh(csm)
        
        final_map = torch.zeros((F, P), device=device)

        # On convertit epsilon en tenseur une fois pour toutes pour éviter les conflits de types
        eps_t = torch.tensor(epsilon, device=device, dtype=csm.dtype)
        sqrt_eps = torch.sqrt(eps_t)

        for f in range(F):
            # Données locales
            a = a_all[f]
            L = L_val[f].unsqueeze(0)   # [1, M]
            U_f = U[f]                  # [M, M]
            
            # 1. Projection
            sumU = a @ U_f 
            abs_sumU2 = torch.abs(sumU)**2 # [P, M]
            
            # 2. Initialisation Lambda (lb)
            # On utilise torch.sqrt et on s'assure que M est un float/tensor
            norm_a = torch.sqrt(torch.tensor(float(M), device=device)) 
            
            # Correction ici : Utilisation de sqrt_eps (Tensor) au lieu de np.sqrt
            lb = (norm_a - sqrt_eps) / (L.max() * sqrt_eps)
            lamb = 0.5 * lb.clamp(min=1e-10) # [P]

            # 3. Newton-Raphson Loop
            for _ in range(20):
                denom = 1 + L * lamb.unsqueeze(1) # [P, M]
                
                f_val = torch.sum(abs_sumU2 / (denom**2), dim=1) - eps_t
                fp_val = -2 * torch.sum((L * abs_sumU2) / (denom**3), dim=1)
                
                lamb = lamb - f_val / fp_val
                lamb = lamb.clamp(min=1e-10)

            # 4. Calcul Puissance (Ta formule MATLAB exacte)
            term_pow = (lamb.unsqueeze(1) / (1 + lamb.unsqueeze(1) * L))**2
            denom_pow = torch.sum(L * abs_sumU2 * term_pow, dim=1)
            
            map_t = 1.0 / torch.clamp(denom_pow, min=1e-12)

            # 5. Correction d'Ambiguïté
            weighting = 1.0 / (1 + L * lamb.unsqueeze(1))
            norm_ac2 = torch.sum(abs_sumU2 * (1 - weighting)**2, dim=1)
            
            final_map[f] = map_t * (norm_ac2 / M)

        return final_map.mean(dim=0) if average else final_map
    
class FB(BaseFrequencySolver):
    def solve(self, csm, freqs, r=1, method='standard', average=True):
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
        a = a / torch.linalg.vector_norm(a)
 
        L, V = torch.linalg.eig(csm)
        if r != 0:
            # On élève chaque valeur propre à la puissance 1/r
            L_pow = L**(1/r)
            
            # 3. Reconstruction de la matrice de poids W
            W = torch.einsum('fmi,fi,fni->fmn', V, L_pow, V.conj())
            
            # 4. Calcul de la forme quadratique a^H * W * a
            quad_form = torch.einsum('fpi,fij,fpj->fp', a.conj(), W, a).real
  
            # 5. Puissance finale element-wise sur l'image
            img = quad_form**r

        else:
            L_log = torch.log(L)
            W = torch.einsum('fmi,fi,fni->fmn', V, L_log, V.conj())
            log_form = torch.einsum('fpi,fij,fpj->fp', a.conj(), W, a).real
            img = torch.exp(log_form)

            

        return img.mean(dim=0) if average else img