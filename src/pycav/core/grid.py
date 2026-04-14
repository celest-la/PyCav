import torch

class Grid:
    def __init__(self, x_coords, z_coords, device="cpu"):
        """
        x_coords: Tenseur 1D des positions en X (ex: linspace)
        z_coords: Tenseur 1D des positions en Z (ex: linspace)
        """
        self.device = torch.device(device)
        self.x = torch.as_tensor(x_coords, dtype=torch.float32, device=self.device)
        self.z = torch.as_tensor(z_coords, dtype=torch.float32, device=self.device)
        
        # Création du maillage (meshgrid)
        # indexing='ij' : la matrice aura pour dimensions (len(x), len(z))
        grid_x, grid_z = torch.meshgrid(self.x, self.z, indexing='ij')
        
        # On stocke les positions sous forme [N_pixels, 3] pour torch.cdist
        self.n_pixels = grid_x.numel()
        self.positions = torch.zeros((self.n_pixels, 3), device=self.device)
        self.positions[:, 0] = grid_x.flatten()
        self.positions[:, 2] = grid_z.flatten() # On utilise Z pour la profondeur
        
        # Utile pour le reshape final lors de l'affichage
        self.shape = (len(self.x), len(self.z))

    @classmethod
    def from_limits(cls, x_lim, z_lim, step, device="cpu"):
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
    def from_verasonics(cls, p_struct, device="cpu"):
        """
        Exemple : crée une grille à partir de la structure P de Verasonics.
        P.startDepth et P.endDepth sont souvent en longueurs d'onde.
        """
        # À adapter selon tes exports, ici on simule la lecture :
        z_start = p_struct.get('startDepth', 0.0)
        z_end = p_struct.get('endDepth', 0.05)
        # On peut fixer un pas par défaut ou le tirer de P
        step = 0.0002 
        
        x_coords = torch.arange(-0.01, 0.01 + step, step)
        z_coords = torch.arange(z_start, z_end + step, step)
        return cls(x_coords, z_coords, device=device)
    


