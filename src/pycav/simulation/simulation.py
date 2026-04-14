import torch as torch
import numpy as np
from pycav.physics.propagation import compute_steering_vectors
from scipy.signal import butter, sosfiltfilt

class getPositions:
    @classmethod
    def random_ellipse(cls,x_position_lim = 0e-3, z_position_lim = 20e-3, x_size_lim = 1e-3, z_size_lim = 1e-3, angle = 0, density=100):

       ### Calculation of the size of the z axis of the ellipse
        def get_val(lim):
            lim=torch.as_tensor(lim)
            if lim.ndim == 0 or lim.shape[0] == 1:
                return lim
            return lim[0] + torch.rand(1) * (lim[1] - lim[0])
        
        x_size=get_val(x_size_lim)
        z_size = get_val(z_size_lim)
       
        #### Calculation of the center of the ellipse (axe z) 
        z_position_lim=torch.as_tensor(z_position_lim)
        if z_position_lim.numel() == 1:    
            z_center=z_position_lim           

        elif z_position_lim.numel() == 2:
            z_center=z_position_lim[0]+z_size/2+torch.rand(1)*(z_position_lim[1]-z_position_lim[0]-2*z_size/2)            
        else:
            raise ValueError("z_position_lim should be of size 1 or 2")

        #### Calculation of the center of the ellipse (axe x) 
        x_position_lim=torch.as_tensor(x_position_lim)
        if x_position_lim.shape[0] == 1:    
            x_center=x_position_lim           

        elif x_position_lim.shape[0] == 2:
            x_center=x_position_lim[0]+x_size/2+torch.rand(1)*(x_position_lim[1]-x_position_lim[0]-2*x_size/2)            
        else:
            raise ValueError("z_position_lim should be of size 1 or 2")    

        center = torch.tensor([x_center, z_center])
        
        ellipse_axes = torch.tensor([x_size, z_size])
        
        number_bubble = int(torch.round(x_size * z_size * density * 1e6).item())
        
        pos_temp=-0.5 * ellipse_axes+ellipse_axes*torch.rand(number_bubble,2)
        
        ind_in=(pos_temp[:,0]**2/(0.5*x_size)**2+pos_temp[:,1]**2/(0.5*z_size)**2)<=1
        pos_temp2 = pos_temp[ind_in,:]

        if angle == None:
            angle = 360 * torch.rand(1)
        angle_rad = torch.as_tensor(angle * torch.pi/180)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # Matrice de rotation 2D
        R = torch.tensor([[cos_a, -sin_a],
            [sin_a, cos_a]])
        bubbles_pos = torch.matmul(pos_temp2, R.t()) + center
        y=torch.zeros(bubbles_pos.shape[0],1)
        bubbles_pos = torch.cat([bubbles_pos, y], dim=1)
        bubbles_pos = bubbles_pos[:, [0, 2, 1]]
        return  bubbles_pos, center, x_size, z_size, angle

class BubbleSource:
    def __init__(self, probe, bubble_positions, c=1540):
        self.probe = probe
        self.c = c
        self.bubble_positions = bubble_positions

class Vokurka(BubbleSource):
    def get_kernel():
        """Cette fonction fourni le kernel de Vokurka"""
        pass

class Stable(BubbleSource):
    def __init__(self, fs, f_hifu, device="cpu"):
        self.fs = fs
        self.f_hifu = f_hifu
        self.device = device

    def get_kernel(self, alpha=1.0, beta=0.8, delta=0.1):
        """
        alpha/beta : asymétrie expansion/compression (génère les harmoniques 2f, 3f)
        delta : différence d'amplitude entre le cycle 1 et le cycle 2 (génère f0/2)
        """
        T = 1 / self.f_hifu
        # On génère exactement 2 cycles
        t = torch.arange(0, 2 * T, 1/self.fs, device=self.device)

        if t.shape[0] % 2 == 0:
            t = torch.arange(0, 2 * T + 1/self.fs, 1/self.fs, device=self.device)


        s = torch.sin(2 * torch.pi * self.f_hifu * t)
        
        # 1. Asymétrie classique (Expansion vs Compression)
        kernel = torch.where(s > 0, s * alpha, s * beta)
        
        # 2. Modulation de période (Sous-harmonique)
        # On réduit un peu l'amplitude du deuxième cycle (t > T)
        mask_cycle2 = (t >= T)
        kernel[mask_cycle2] *= (1 - delta)
        
        # Fenêtrage
        window = torch.hann_window(len(t), periodic=False, device=self.device)
        return (kernel * window) / torch.norm(kernel * window)

class StableSimulator:
    def __init__(self, probe, bubble_positions, c0=1540, fhifu=1e6, device="cpu"):
        self.probe = probe
        self.bubble_positions = bubble_positions  # (Nb, 3) ou (Nb, 2)
        self.c0 = c0
        self.fhifu = fhifu
        self.nb = bubble_positions.shape[0]
        self.device = device

    def _compute_batch_delays(self, batch_pos, t0):
        """Calcule les index de délais pour un groupe de bulles."""
        # dist: (Nelem, Batch_Nb)
        dist = torch.cdist(self.probe.positions, batch_pos, p=2)
        return (dist / self.c0) * self.probe.fs + t0

    def _get_source_events(self, n_cycles, batch_nb, A0, sd, subharmonic=True):
        # Si subharmonic, on veut une impulsion tous les 2 cycles HIFU
        step = 2 if subharmonic else 1
        
        T_samples = self.probe.fs / self.fhifu
        
        # On crée les indices de temps : 0.5, 2.5, 4.5... au lieu de 0.5, 1.5, 2.5...
        t_emit = (torch.arange(0, n_cycles, step, device=self.device).float() + 0.5) * T_samples
        
        # Le nombre d'amplitudes doit correspondre au nombre de Diracs générés
        n_pulses = t_emit.shape[0]
        amps = A0 + sd * torch.randn(n_pulses, batch_nb, device=self.device)
        
        return t_emit, amps

    def _splat_linear(self, rf_grid, t_arrival, amplitudes):
        """Projette les impacts sur la grille RF (Samples x Nelem)."""
        n_samples, nelem = rf_grid.shape
        
        # 1. Flattening pour traitement vectorisé
        t_flat = t_arrival.reshape(-1)
        a_flat = amplitudes.reshape(-1)
        
        # indices d'éléments répétés pour chaque impact
        # t_arrival est (nelem, n_cycles, batch_nb)
        elem_idx = torch.arange(nelem, device=self.device).view(-1, 1, 1).expand_as(t_arrival).reshape(-1)

        t_floor = torch.floor(t_flat).long()
        frac = t_flat - t_floor.float()

        # 2. Accumulation sur les deux samples adjacents
        for offset, weight in [(0, 1 - frac), (1, frac)]:
            idx = t_floor + offset
            mask = (idx >= 0) & (idx < n_samples)
            
            rf_grid.index_put_(
                (idx[mask], elem_idx[mask]), 
                (a_flat * weight)[mask], 
                accumulate=True
            )

    def get_delayed_dirac(self, n_cycles, A0=1, t0=0, t_acq=100e-6, sd=0.1, batch_size=1000):
        """Fonction principale avec gestion du batching."""
        nelem = self.probe.positions.shape[0]
        n_samples = int(t_acq * self.probe.fs)
        dirac_RF = torch.zeros(n_samples, nelem, device=self.device)

        for i in range(0, self.nb, batch_size):
            # --- 1. Sélection du Batch ---
            batch_pos = self.bubble_positions[i : i + batch_size]
            curr_batch_nb = batch_pos.shape[0]
            
            # --- 2. Géométrie & Source ---
            index_tau = self._compute_batch_delays(batch_pos, t0)
            index_dirac, batch_amps = self._get_source_events(n_cycles, curr_batch_nb, A0, sd)
            
            # --- 3. Combinaison des temps (Broadcast) ---
            # (nelem, batch, 1) + (1, 1, n_cycles) -> (nelem, n_cycles, batch)
            tau_final = (index_tau.unsqueeze(-1) + index_dirac.view(1, 1, -1)).permute(0, 2, 1)
            
            # --- 4. Expansion des amplitudes ---
            # On duplique les amplitudes de la bulle pour chaque élément de la sonde
            # (n_cycles, batch) -> (nelem, n_cycles, batch)
            full_amps = batch_amps.unsqueeze(0).expand_as(tau_final)
            
            # --- 5. Rendu sur la grille ---
            self._splat_linear(dirac_RF, tau_final, full_amps)
            
        return dirac_RF


    def apply_kernel(self, dirac_rf, kernel):
        """
        Applique le kernel de cavitation sur la matrice de Diracs.
        dirac_rf : Tensor (n_samples, nelem)
        kernel : Tensor (kernel_length,)
        """
        # 1. Préparation des dimensions (Batch, Channels, Length)
        # On transpose pour avoir les éléments en 'Channels'
        x = dirac_rf.t().unsqueeze(0) # (1, nelem, n_samples)
        
        # 2. Préparation du Kernel
        # On doit le répéter pour chaque canal (nelem)
        # Shape finale : (nelem, 1, kernel_length)
        nelem = dirac_rf.shape[1]
        k_size = kernel.shape[0]
        weight = kernel.view(1, 1, -1).repeat(nelem, 1, 1)
        
        # 3. Convolution
        # 'padding=same' permet de garder la même longueur de signal
        # 'groups=nelem' permet de convoluer chaque channel avec son propre kernel
        rf_conv = torch.nn.functional.conv1d(x, weight, padding='same', groups=nelem)
        
        # 4. Retour au format original (n_samples, nelem)
        return rf_conv.squeeze(0).t()

    

    def apply_probe_filter(self, rf_tensor):
        """
        Applique la réponse fréquentielle de la sonde L7-5.
        f_center : 5.5 MHz
        """
        band = self.probe.fc
        fs = self.probe.fs
        nyq = 0.5 * fs
        
        # Normalisation par rapport à Nyquist
        low = band[0] / nyq
        high = min(band[1] / nyq, 0.95)
        
        sos = butter(4, [low, high], btype='band', output='sos')
        
        rf_np = rf_tensor.cpu().numpy()
        # filtfilt est impératif pour ne pas décaler tes fronts de montée (phase linéaire)
        rf_filt = sosfiltfilt(sos, rf_np, axis=0)
        
        return torch.from_numpy(rf_filt.copy()).to(rf_tensor.device)

class BBagSimulator():
        def __init__(self, probe, bubble_positions, c0=1540, fhifu=1e6, device="cpu"):
            self.probe = probe
            self.bubble_positions = bubble_positions  # (Nb, 3) ou (Nb, 2)
            self.c0 = c0
            self.fhifu = fhifu
            self.nb = bubble_positions.shape[0]
            self.device = device
        def get_RF(self, n_cycles=200, A0=1, t0=0, t_acq=100e-6, batch_size=1000):
            source_signals=A0 * torch.rand((int(n_cycles*self.probe.fs/self.fhifu), self.nb),device=self.device)
            n_bin = n_bin = int(t_acq * self.probe.fs)
            source_fft = torch.fft.rfft(source_signals, dim=0, n=n_bin)
            freq=torch.fft.rfftfreq(n_bin, d=1/self.probe.fs, device=self.device)
            steer_vec = compute_steering_vectors(probe=self.probe, grid_positions=self.bubble_positions, freqs=freq, c0=self.c0)
            delayed_fft = torch.einsum('fb,fbe->fe', source_fft, steer_vec)
            return torch.fft.irfft(delayed_fft, n = n_bin, dim = 0)
        
        def apply_probe_filter(self, rf_tensor):
            """
            Applique la réponse fréquentielle de la sonde L7-5.
            f_center : 5.5 MHz
            """
            band = self.probe.fc
            fs = self.probe.fs
            nyq = 0.5 * fs
            
            # Normalisation par rapport à Nyquist
            low = band[0] / nyq
            high = min(band[1] / nyq, 0.95)
            
            sos = butter(2, [low, high], btype='band', output='sos')
            
            rf_np = rf_tensor.cpu().numpy()
            # filtfilt est impératif pour ne pas décaler tes fronts de montée (phase linéaire)
            rf_filt = sosfiltfilt(sos, rf_np, axis=0)
            
            return torch.from_numpy(rf_filt.copy()).to(rf_tensor.device)