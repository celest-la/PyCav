import scipy.io as sio
import torch
import numpy as np

def load_verasonics_data(file_path):
    """
    Charge un fichier .mat exporté par Verasonics.
    Retourne un dictionnaire avec les objets convertis en Python/Numpy.
    """
    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    
    # mat_data contient généralement : Trans, P, RcvData, etc.
    return mat_data

def get_rf_data(mat_data, buffer_idx=0):
    """
    Extrait les signaux RF du buffer RcvData.
    Retourne un tenseur [Nt, Ne].
    """
    # Verasonics stocke souvent les données en int16 dans une liste de tableaux
    rf = mat_data['RcvData'][buffer_idx]
    return torch.from_numpy(rf.astype(np.float32))