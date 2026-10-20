# src/py_cav/core/probe.py
from __future__ import annotations
from typing import Union, Tuple, Optional, Any

import torch
import numpy as np
import scipy.io as sio
from pycav.utils import validation_probe


class Probe:
    @validation_probe
    def __init__(self, positions: Union[torch.Tensor, list, np.ndarray], fs: float, fc: torch.Tensor, device: Union[str, torch.device]="cpu") -> None:
        self.device = torch.device(device)
        self.positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        self.fs: float = float(fs)
        self.fc: torch.Tensor = fc

    @property
    def n_elements(self):
        return self.positions.shape[0]
    
    @classmethod
    def from_verasonics(cls, trans_struct: dict[str, Any], resource_struct: dict[str, Any], device: Union[str, torch.device] ="cpu") -> "Probe":
        """
        A faire : Crée une sonde à partir des structures MATLAB Verasonics.  """        
        return cls(positions=pos_m, fs=fs, device=device)
    
    
    @classmethod
    def from_linear(cls, n_el: int = 128, pitch: float = 0.3, fs: float = 22e6, device: Union[str, torch.device] ="cpu",fc: Optional[torch.Tensor] = None) -> "Probe":
        """
        Crée une sonde linéaire centrée sur x=0.
        n_el  : Nombre d'éléments (ex: 128)
        pitch : Espacement entre les centres des éléments en mètres (ex: 0.3e-3)
        fs    : Fréquence d'échantillonnage (par défaut 22MHz)
        """
        if fc is None:
            fc = torch.tensor([4.5e6, 7.5e6], dtype=torch.float32) # ICI
        # Calcul des positions en X pour que la sonde soit centrée
        # (Ex: pour 2 éléments à pitch 0.3, on aura -0.15 et +0.15)
        x = (torch.arange(n_el) - (n_el - 1) / 2) * pitch
        
        # On crée la matrice des positions [N_elements, 3] (X, Y, Z)
        positions = torch.zeros((n_el, 3), device=device)
        positions[:, 0] = x
        # Y et Z restent à 0 pour une sonde linéaire dans le plan XZ
        
        return cls(positions=positions, fs=fs, fc=fc, device=device)

    @classmethod
    def from_3d_coordinates(
        cls, 
        xp: Union[list, np.ndarray, torch.Tensor], 
        yp: Union[list, np.ndarray, torch.Tensor], 
        zp: Union[list, np.ndarray, torch.Tensor], 
        fs: float, 
        fc: Optional[torch.Tensor] = None, 
        device: Union[str, torch.device] = "cpu"
    ) -> "Probe":
        """
        Crée une sonde 3D à partir de vecteurs de coordonnées (xp, yp, zp).
        """
        device = torch.device(device)
        xp_t = torch.as_tensor(xp, dtype=torch.float32, device=device).flatten()
        yp_t = torch.as_tensor(yp, dtype=torch.float32, device=device).flatten()
        zp_t = torch.as_tensor(zp, dtype=torch.float32, device=device).flatten()
        
        if not (len(xp_t) == len(yp_t) == len(zp_t)):
            raise ValueError("xp, yp, zp doivent avoir la même longueur.")
            
        positions = torch.stack([xp_t, yp_t, zp_t], dim=1)  # [Ne, 3]
        
        if fc is None:
            fc = torch.tensor([1.5e6, 5.0e6], dtype=torch.float32, device=device)
        elif not torch.is_tensor(fc):
            fc = torch.as_tensor(fc, dtype=torch.float32, device=device)
            
        return cls(positions=positions, fs=fs, fc=fc, device=device)

    @classmethod
    def from_param_dict(cls, param: dict[str, Any], device: Union[str, torch.device] = "cpu") -> "Probe":
        """
        Crée une sonde directement depuis le dictionnaire MATLAB `param`.
        """
        return cls.from_3d_coordinates(
            xp=param["xp"],
            yp=param["yp"],
            zp=param["zp"],
            fs=param["fs"],
            fc=param.get("fc", [1.5e6, 5.0e6]),
            device=device
        )

    @classmethod
    def from_matlab_file(cls, filepath: Union[str, Path], struct_name: str = "param", device: Union[str, torch.device] = "cpu") -> "Probe":
        """
        Loads a .mat file directly and constructs the Probe.
        """
        import scipy.io as sio
        mat_data = sio.loadmat(filepath, squeeze_me=True)
        raw = mat_data[struct_name]
        
        param = {}
        for name in raw.dtype.names:
            val = raw[name]
            param[name] = val.item() if getattr(val, 'shape', None) == () else val
            
        return cls.from_param_dict(param, device=device)