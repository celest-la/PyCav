from __future__ import annotations
from typing import Union, Tuple, Optional, List

import torch
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.physics.propagation import compute_steering_vectors

#from pycav.optim.operators import CmfOperator, WeightedGradient, BlockOperator
#from pycav.optim.functionals import L2DataFidelity, BlockFunctional
#from pycav.optim.priors import L1Prior, IsotropicTV, AnisotropicTV
#from pycav.optim.solvers import PDHGSolver, AdaptivePDHGSolver


class BaseFrequencySolver:
    def __init__(self, probe: Probe, grid: Grid, c: float = 1540) -> None:
        self.probe = probe
        self.grid = grid
        self.c = c

    @torch.no_grad()
    def get_steering_vectors(self, freqs: Union[torch.Tensor, List[float], float]) -> torch.Tensor:
        return compute_steering_vectors(self.probe, self.grid.positions, freqs, self.c)

    def _format_output(self, img: torch.Tensor, average: bool) -> torch.Tensor:
        F = img.shape[0]
        img_reshaped = img.view(F, *self.grid.shape)
        if average:
            return img_reshaped.mean(dim=0)
        return img_reshaped


class DAS(BaseFrequencySolver):
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], average: bool = True) -> torch.Tensor:
        a = self.get_steering_vectors(freqs)
        a = a / torch.tensor(self.probe.positions.shape[0])
        img = torch.einsum('fpi,fij,fpj->fp', a.conj(), csm, a).real
        return self._format_output(img, average)


class RCB(BaseFrequencySolver):
    @torch.no_grad()
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], epsilon: float = 0.1, average: bool = True) -> torch.Tensor:
        a_all = self.get_steering_vectors(freqs)
        F, P, M = a_all.shape
        device = csm.device
        
        L_val, U = torch.linalg.eigh(csm)
        L_val = torch.clamp(L_val, min=1e-12)
        
        final_map = torch.zeros((F, P), device=device, dtype=torch.float32)
        eps_t = torch.tensor(epsilon, device=device, dtype=torch.float32)
        sqrt_eps = torch.sqrt(eps_t)

        for f in range(F):
            a = a_all[f]
            L = L_val[f].unsqueeze(0)
            U_f = U[f]
            
            sumU = a.conj() @ U_f
            abs_sumU2 = torch.abs(sumU)**2
            norm_a = torch.sqrt(torch.tensor(float(M), device=device, dtype=torch.float32)) 
            
            lb = (norm_a - sqrt_eps) / (L.max() * sqrt_eps)
            lamb_init = 0.5 * max(lb.item(), 1e-10)
            lamb = torch.full((P,), lamb_init, device=device, dtype=torch.float32)

            for _ in range(20):
                denom = 1 + L * lamb.unsqueeze(1)   
                f_val = torch.sum(abs_sumU2 / (denom**2), dim=1) - eps_t
                fp_val = -2 * torch.sum((L * abs_sumU2) / (denom**3), dim=1)
                
                lamb = lamb - f_val / fp_val
                lamb = lamb.clamp(min=1e-10)

            term_pow = (lamb.unsqueeze(1) / (1 + lamb.unsqueeze(1) * L))**2
            denom_pow = torch.sum(L * abs_sumU2 * term_pow, dim=1)
            map_t = 1.0 / torch.clamp(denom_pow, min=1e-12)

            weighting = 1.0 / (1 + L * lamb.unsqueeze(1))
            norm_ac2 = torch.sum(abs_sumU2 * (1 - weighting)**2, dim=1)
            final_map[f] = map_t * (norm_ac2 / M)

        return self._format_output(final_map, average)


class FB(BaseFrequencySolver):
    @torch.no_grad()
    def solve(self, csm: torch.Tensor, freqs: Union[torch.Tensor, List[float], float], r: float = 1, method: str = 'standard', average: bool = True) -> torch.Tensor:
        a = self.get_steering_vectors(freqs)
        norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
        a = a / norm
        L, V = torch.linalg.eigh(csm)
        L = torch.clamp(L, min=1e-12)
        proj = torch.einsum('fpm,fmi->fpi', a.conj(), V)
        if r != 0:
            L_pow = L**(1/r)
            quad_form = torch.einsum('fpi,fi,fpi->fp', proj, L_pow, proj.conj()).real
            img = (quad_form**r) / self.probe.positions.shape[0]
        else:
            L_log = torch.log(L)
            W = torch.einsum('fmi,fi,fni->fmn', V, L_log, V.conj())
            log_form = torch.einsum('fpi,fij,fpj->fp', a.conj(), W, a).real
            img = torch.exp(log_form) / self.probe.positions.shape[0]           

        return self._format_output(img, average)
