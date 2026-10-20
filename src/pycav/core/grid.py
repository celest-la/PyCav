from __future__ import annotations
from typing import Union, Tuple, Optional

import torch
class Grid:
    def __init__(
        self, 
        x_coords: Union[torch.Tensor, list, float], 
        z_coords: Union[torch.Tensor, list, float], 
        y_coords: Optional[Union[torch.Tensor, list, float]] = None, 
        device="cpu") -> None:
        self.device = torch.device(device)
        self.x = torch.as_tensor(x_coords, dtype=torch.float32, device=self.device)
        self.z = torch.as_tensor(z_coords, dtype=torch.float32, device=self.device)
        
        if y_coords is not None:
            self.y = torch.as_tensor(y_coords, dtype=torch.float32, device=self.device)
            # 3D Meshgrid [Nx, Ny, Nz]
            grid_x, grid_y, grid_z = torch.meshgrid(self.x, self.y, self.z, indexing='ij')
            
            self.n_pixels = grid_x.numel()
            self.positions = torch.zeros((self.n_pixels, 3), device=self.device)
            self.positions[:, 0] = grid_x.flatten()
            self.positions[:, 1] = grid_y.flatten()
            self.positions[:, 2] = grid_z.flatten()
            
            self.shape = (len(self.x), len(self.y), len(self.z))
            self.is_3d = True
        else:
            self.y = torch.zeros(1, device=self.device)
            grid_x, grid_z = torch.meshgrid(self.x, self.z, indexing='ij')
            
            self.n_pixels = grid_x.numel()
            self.positions = torch.zeros((self.n_pixels, 3), device=self.device)
            self.positions[:, 0] = grid_x.flatten()
            self.positions[:, 1] = 0.0
            self.positions[:, 2] = grid_z.flatten()
            
            self.shape = (len(self.x), len(self.z))
            self.is_3d = False

    @classmethod
    def from_limits(cls, x_lim: Tuple[float, float], z_lim: Tuple[float, float], step: float, device="cpu") -> "Grid":
        """
        Crée une grille uniforme à partir de bornes.
        x_lim: (min, max) en mètres
        z_lim: (min, max) en mètres
        step: pas entre pixels en mètres
        """
        x_coords = torch.arange(x_lim[0], x_lim[1] + step, step)
        z_coords = torch.arange(z_lim[0], z_lim[1] + step, step)
        return cls(x_coords, z_coords, device=device)

    @classmethod
    def from_verasonics(cls, p_struct: dict[str, Any], device: Union[str, torch.device] ="cpu") ->"Grid":
        """
        Fonction à compléter et tester
        """
        return cls(x_coords, z_coords, device=device)
    
    @classmethod
    def from_limits_3d(
        cls, 
        x_lim: Tuple[float, float], 
        y_lim: Tuple[float, float], 
        z_lim: Tuple[float, float], 
        step: float, 
        device="cpu"
    ) -> "Grid":
        """
        Crée une grille 3D uniforme [Nx x Ny x Nz].
        """
        x_coords = torch.arange(x_lim[0], x_lim[1] + step, step)
        y_coords = torch.arange(y_lim[0], y_lim[1] + step, step)
        z_coords = torch.arange(z_lim[0], z_lim[1] + step, step)
        return cls(x_coords=x_coords, z_coords=z_coords, y_coords=y_coords, device=device)


