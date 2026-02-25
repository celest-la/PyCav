# src/py_cav/core/probe.py
import torch
import numpy as np
import scipy.io as sio
from pycav.utils import validation_probe

class Probe:
    @validation_probe
    def __init__(self, positions, fs, fc,device="cpu"):
        self.device = torch.device(device)
        self.positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        self.fs: float = float(fs)
        self.fc: torch.tensor = fc

    @property
    def n_elements(self):
        return self.positions.shape[0]
    
    @classmethod
    def from_matlab(cls, filepath, device="cpu"):
        # 1. Chargement du fichier .mat
        data = sio.loadmat(filepath)
        
        # 2. Extraction des données (selon ta structure MATLAB)
        # Supposons que ta structure s'appelle 'Sonde' dans MATLAB
        mat_struct = data['Sonde'][0,0]
        pos_raw = mat_struct['pos'] # Matrice [N x 3]
        fs_raw = mat_struct['fs'][0,0] # Scalaire
        
        # 3. L'INSTANCIATION : On crée l'objet en appelant cls(...)
        # C'est comme si on faisait Probe(pos_raw, fs_raw, device)
        return cls(positions=pos_raw, fs=fs_raw, device=device)
    
   # Dans src/pycav/core/probe.py

    @classmethod
    def from_verasonics(cls, trans_struct, resource_struct, device="cpu"):
        """
        trans_struct : objet Trans de MATLAB
        resource_struct : objet Resource de MATLAB (pour fs)
        """
        # 1. Récupération des positions [M, 3]
        # Trans.ElementPos est souvent en [x, y, z, beam_x, beam_y, beam_z]
        # On ne garde que les 3 premières colonnes (x, y, z)
        pos_mm = torch.from_numpy(trans_struct.ElementPos[:, :3])
        pos_m = pos_mm * 1e-3 # Si c'était en mm
        
        # 2. Récupération de la fréquence d'échantillonnage
        # Resource.Parameters.fS est en MHz
        fs_mhz = resource_struct.Parameters.fS
        fs = fs_mhz * 1e6
        
        return cls(positions=pos_m, fs=fs, device=device)
    
    
    @classmethod
    def from_linear(cls, n_el: int = 128, pitch: float = 0.3, fs: float = 22e6, device="cpu",fc: torch.tensor = torch.tensor([4.5*1e6,7.5*1e6])):
        """
        Crée une sonde linéaire centrée sur x=0.
        n_el  : Nombre d'éléments (ex: 128)
        pitch : Espacement entre les centres des éléments en mètres (ex: 0.3e-3)
        fs    : Fréquence d'échantillonnage (par défaut 22MHz)
        """
        # Calcul des positions en X pour que la sonde soit centrée
        # (Ex: pour 2 éléments à pitch 0.3, on aura -0.15 et +0.15)
        x = (torch.arange(n_el) - (n_el - 1) / 2) * pitch
        
        # On crée la matrice des positions [N_elements, 3] (X, Y, Z)
        positions = torch.zeros((n_el, 3), device=device)
        positions[:, 0] = x
        # Y et Z restent à 0 pour une sonde linéaire dans le plan XZ
        
        return cls(positions=positions, fs=fs, fc=fc, device=device)