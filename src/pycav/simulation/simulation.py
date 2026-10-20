from __future__ import annotations
from typing import Union, Tuple, Optional

import torch
import numpy as np
from pycav.physics.propagation import compute_steering_vectors
from pycav.core.probe import Probe


class getPositions:
    @classmethod
    @torch.no_grad()
    def random_ellipse(
        cls,
        x_position_lim: Union[float, Tuple[float, float], list, torch.Tensor] = 0e-3,
        z_position_lim: Union[float, Tuple[float, float], list, torch.Tensor] = 20e-3,
        x_size_lim: Union[float, Tuple[float, float], list, torch.Tensor] = 1e-3,
        z_size_lim: Union[float, Tuple[float, float], list, torch.Tensor] = 1e-3,
        angle: Optional[float] = 0,
        density: int = 100,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[float, torch.Tensor]]:
        """Génère un nuage 2D de positions de bulles dans le plan X-Z (y=0)."""
        device = torch.device(device)

        def get_val(lim):
            lim_t = torch.as_tensor(lim, device=device, dtype=torch.float32)
            if lim_t.ndim == 0 or lim_t.shape[0] == 1:
                return lim_t.squeeze()
            return lim_t[0] + torch.rand(1, device=device) * (lim_t[1] - lim_t[0])

        x_size = get_val(x_size_lim)
        z_size = get_val(z_size_lim)

        z_position_lim_t = torch.as_tensor(z_position_lim, device=device, dtype=torch.float32)
        if z_position_lim_t.numel() == 1:
            z_center = z_position_lim_t
        elif z_position_lim_t.numel() == 2:
            z_center = z_position_lim_t[0] + z_size / 2 + torch.rand(1, device=device) * (
                z_position_lim_t[1] - z_position_lim_t[0] - z_size
            )
        else:
            raise ValueError("z_position_lim should be of size 1 or 2")

        x_position_lim_t = torch.as_tensor(x_position_lim, device=device, dtype=torch.float32)
        if x_position_lim_t.numel() == 1:
            x_center = x_position_lim_t
        elif x_position_lim_t.numel() == 2:
            x_center = x_position_lim_t[0] + x_size / 2 + torch.rand(1, device=device) * (
                x_position_lim_t[1] - x_position_lim_t[0] - x_size
            )
        else:
            raise ValueError("x_position_lim should be of size 1 or 2")

        center = torch.tensor([x_center, z_center], dtype=torch.float32, device=device)
        ellipse_axes = torch.tensor([x_size, z_size], dtype=torch.float32, device=device)

        number_bubble = int(torch.round(x_size * z_size * density * 1e6).item())

        pos_temp = -0.5 * ellipse_axes + ellipse_axes * torch.rand((number_bubble, 2), device=device)

        ind_in = (pos_temp[:, 0]**2 / (0.5 * x_size)**2 + pos_temp[:, 1]**2 / (0.5 * z_size)**2) <= 1
        pos_temp2 = pos_temp[ind_in, :]

        if angle is None:
            angle = 360 * torch.rand(1, device=device)
        angle_rad = torch.as_tensor(angle * torch.pi / 180, device=device)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32, device=device)
        bubbles_pos = torch.matmul(pos_temp2, R.t()) + center
        y = torch.zeros((bubbles_pos.shape[0], 1), device=device)
        bubbles_pos = torch.cat([bubbles_pos, y], dim=1)
        bubbles_pos = bubbles_pos[:, [0, 2, 1]]  # [X, Y, Z]

        return bubbles_pos, center, x_size, z_size, angle

    @classmethod
    @torch.no_grad()
    def random_ellipsoid(
        cls,
        x_position_lim: Union[float, Tuple[float, float]] = 0e-3,
        y_position_lim: Union[float, Tuple[float, float]] = 0e-3,
        z_position_lim: Union[float, Tuple[float, float]] = 20e-3,
        x_size_lim: Union[float, Tuple[float, float]] = 1e-3,
        y_size_lim: Union[float, Tuple[float, float]] = 1e-3,
        z_size_lim: Union[float, Tuple[float, float]] = 1e-3,
        density: int = 100,
        num_bubbles: Optional[int] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Génère un nuage 3D de positions de bulles distribuées uniformément dans un ellipsoïde [X, Y, Z].
        density est exprimée en bulles / mm^3.
        """
        device = torch.device(device)

        def get_val(lim):
            lim_t = torch.as_tensor(lim, device=device, dtype=torch.float32)
            if lim_t.numel() == 1:
                return lim_t.squeeze()
            return lim_t[0] + torch.rand(1, device=device) * (lim_t[1] - lim_t[0])

        x_size = get_val(x_size_lim)
        y_size = get_val(y_size_lim)
        z_size = get_val(z_size_lim)

        x_center = get_val(x_position_lim)
        y_center = get_val(y_position_lim)
        z_center = get_val(z_position_lim)

        center = torch.stack([x_center, y_center, z_center])
        axes = torch.stack([x_size, y_size, z_size])

        if num_bubbles is None:
            # Volume d'un ellipsoïde 3D = (4/3) * pi * a * b * c (en m^3)
            volume_m3 = (4.0 / 3.0) * torch.pi * (0.5 * x_size) * (0.5 * y_size) * (0.5 * z_size)
            # Conversion m^3 en mm^3 (facteur 1e9)
            number_bubble = int(torch.round(volume_m3 * density * 1e9).item())
            number_bubble = max(number_bubble, 1)
        else:
            number_bubble = int(num_bubbles)

        # Rejection sampling inside 3D ellipsoid
        pos_temp = -0.5 * axes + axes * torch.rand((number_bubble * 4, 3), device=device)
        norm_sq = (pos_temp[:, 0] / (0.5 * x_size))**2 + (pos_temp[:, 1] / (0.5 * y_size))**2 + (pos_temp[:, 2] / (0.5 * z_size))**2
        valid_pos = pos_temp[norm_sq <= 1.0][:number_bubble]

        bubbles_pos = valid_pos + center
        return bubbles_pos, center, axes


class BubbleSource:
    def __init__(self, probe: Probe, bubble_positions: torch.Tensor, c: float = 1540) -> None:
        self.probe = probe
        self.c = c
        self.bubble_positions = bubble_positions


class Vokurka(BubbleSource):
    def get_kernel(self):
        """Cette fonction fournit le kernel de Vokurka"""
        pass


class Stable(BubbleSource):
    def __init__(self, fs: float, f_hifu: float, device: Union[str, torch.device] = "cpu") -> None:
        self.fs = fs
        self.f_hifu = f_hifu
        self.device = torch.device(device)

    @torch.no_grad()
    def get_kernel(self, alpha: float = 1.0, beta: float = 0.8, delta: float = 0.1) -> torch.Tensor:
        T = 1 / self.f_hifu
        t = torch.arange(0, 2 * T, 1/self.fs, device=self.device, dtype=torch.float32)

        if t.shape[0] % 2 == 0:
            t = torch.arange(0, 2 * T + 1/self.fs, 1/self.fs, device=self.device, dtype=torch.float32)

        s = torch.sin(2 * torch.pi * self.f_hifu * t)
        kernel = torch.where(s > 0, s * alpha, s * beta)
        
        mask_cycle2 = (t >= T)
        kernel[mask_cycle2] *= (1 - delta)
        
        window = torch.hann_window(len(t), periodic=False, device=self.device, dtype=torch.float32)
        return (kernel * window) / torch.norm(kernel * window)


class StableSimulator:
    def __init__(self, probe: Probe, bubble_positions: torch.Tensor, c0: float = 1540, fhifu: float = 1e6, device: Union[str, torch.device] = "cpu") -> None:
        self.device = torch.device(device)
        self.probe = probe
        self.bubble_positions = bubble_positions.to(self.device)  # (Nb, 3)
        self.c0 = c0
        self.fhifu = fhifu
        self.nb = bubble_positions.shape[0]

    @torch.no_grad()
    def _compute_batch_delays(self, batch_pos: torch.Tensor, t0: float = 0) -> torch.Tensor:
        dist = torch.cdist(self.probe.positions.to(self.device), batch_pos, p=2)
        return (dist / self.c0) * self.probe.fs + t0

    @torch.no_grad()
    def _get_source_events(self, n_cycles: int, batch_nb: int, A0: float, sd: float, subharmonic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        step = 2 if subharmonic else 1
        T_samples = self.probe.fs / self.fhifu
        t_emit = (torch.arange(0, n_cycles, step, device=self.device, dtype=torch.float32).float() + 0.5) * T_samples
        n_pulses = t_emit.shape[0]
        amps = A0 + sd * torch.randn(n_pulses, batch_nb, device=self.device, dtype=torch.float32)
        return t_emit, amps

    @torch.no_grad()
    def _splat_linear(self, rf_grid: torch.Tensor, t_arrival: torch.Tensor, amplitudes: torch.Tensor) -> None:
        n_samples, nelem = rf_grid.shape
        t_flat = t_arrival.reshape(-1)
        a_flat = amplitudes.reshape(-1)
        
        elem_idx = torch.arange(nelem, device=self.device).view(-1, 1, 1).expand_as(t_arrival).reshape(-1)
        t_floor = torch.floor(t_flat).long()
        frac = t_flat - t_floor.float()

        for offset, weight in [(0, 1 - frac), (1, frac)]:
            idx = t_floor + offset
            mask = (idx >= 0) & (idx < n_samples)
            rf_grid.index_put_(
                (idx[mask], elem_idx[mask]), 
                (a_flat * weight)[mask], 
                accumulate=True
            )

    @torch.no_grad()
    def get_delayed_dirac(self, n_cycles: int, A0: float = 1, t0: float = 0, t_acq: float = 100e-6, sd: float = 0.1, batch_size: int = 1000) -> torch.Tensor:
        nelem = self.probe.positions.shape[0]
        n_samples = int(t_acq * self.probe.fs)
        dirac_RF = torch.zeros(n_samples, nelem, device=self.device)

        for i in range(0, self.nb, batch_size):
            batch_pos = self.bubble_positions[i : i + batch_size]
            curr_batch_nb = batch_pos.shape[0]
            
            index_tau = self._compute_batch_delays(batch_pos, t0)
            index_dirac, batch_amps = self._get_source_events(n_cycles, curr_batch_nb, A0, sd)
            
            tau_final = (index_tau.unsqueeze(-1) + index_dirac.view(1, 1, -1)).permute(0, 2, 1)
            full_amps = batch_amps.unsqueeze(0).expand_as(tau_final)
            
            self._splat_linear(dirac_RF, tau_final, full_amps)
            
        return dirac_RF

    @torch.no_grad()
    def apply_kernel(self, dirac_rf: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        x = dirac_rf.t().unsqueeze(0)
        nelem = dirac_rf.shape[1]
        weight = kernel.view(1, 1, -1).repeat(nelem, 1, 1)
        rf_conv = torch.nn.functional.conv1d(x, weight, padding='same', groups=nelem)
        return rf_conv.squeeze(0).t()

    def apply_probe_filter(self, rf_tensor: torch.Tensor) -> torch.Tensor:
        n = rf_tensor.shape[0]
        fft_rf = torch.fft.rfft(rf_tensor, dim=0)
        freqs = torch.fft.rfftfreq(n, d=1/self.probe.fs, device=self.device)
        
        band = self.probe.fc.to(self.device)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        fft_rf[~mask, :] = 0
        
        return torch.fft.irfft(fft_rf, n=n, dim=0)

    @torch.no_grad()
    def simulate(
        self, n_cycles: int = 200, t_acq: float = 100e-6, A0: float = 1, sd: float = 0.1, 
        alpha: float = 1.0, beta: float = 0.5, delta: float = 0.2, 
        noise_level: float = 0.001, apply_filter: bool = True, batch_size: int = 1000
    ) -> torch.Tensor:
        diracs = self.get_delayed_dirac(n_cycles=n_cycles, A0=A0, t_acq=t_acq, sd=sd, batch_size=batch_size)
        k_sub = Stable(self.probe.fs, self.fhifu, device=self.device).get_kernel(alpha=alpha, beta=beta, delta=delta)
        rf = self.apply_kernel(diracs, k_sub)
        
        if noise_level > 0:
            noise = noise_level * torch.randn_like(rf) * rf.max()
            rf = rf + noise
            
        if apply_filter:
            rf = self.apply_probe_filter(rf)
            
        return rf


class BBagSimulator:
    def __init__(self, probe: Probe, bubble_positions: torch.Tensor, c0: float = 1540, fhifu: float = 1e6, device: Union[str, torch.device] = "cpu") -> None:
        self.device = torch.device(device)
        self.probe = probe
        self.bubble_positions = bubble_positions.to(self.device)  # (Nb, 3)
        self.c0 = c0
        self.fhifu = fhifu
        self.nb = bubble_positions.shape[0]

    @torch.no_grad()
    def get_RF(self, n_cycles: int = 200, A0: float = 1, t0: float = 0, t_acq: float = 100e-6, batch_size: int = 1000) -> torch.Tensor:
        n_bin = int(t_acq * self.probe.fs)
        n_source_samples = int(n_cycles * self.probe.fs / self.fhifu)
        freq = torch.fft.rfftfreq(n_bin, d=1/self.probe.fs, device=self.device)
        
        delayed_fft = torch.zeros((len(freq), self.probe.n_elements), dtype=torch.complex64, device=self.device)
        
        for i in range(0, self.nb, batch_size):
            batch_pos = self.bubble_positions[i : i + batch_size]
            curr_batch_nb = batch_pos.shape[0]
            
            source_signals = A0 * torch.rand((n_source_samples, curr_batch_nb), device=self.device)
            source_fft = torch.fft.rfft(source_signals, dim=0, n=n_bin)
            
            steer_vec = compute_steering_vectors(probe=self.probe, grid_positions=batch_pos, freqs=freq, c0=self.c0)
            delayed_fft += torch.einsum('fb, fbe -> fe', source_fft, steer_vec)
            
        return torch.fft.irfft(delayed_fft, n=n_bin, dim=0)

    def apply_probe_filter(self, rf_tensor: torch.Tensor) -> torch.Tensor:
        n = rf_tensor.shape[0]
        fft_rf = torch.fft.rfft(rf_tensor, dim=0)
        freqs = torch.fft.rfftfreq(n, d=1/self.probe.fs, device=self.device)
        
        band = self.probe.fc.to(self.device)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        fft_rf[~mask, :] = 0
        
        return torch.fft.irfft(fft_rf, n=n, dim=0)

    @torch.no_grad()
    def simulate(
        self, t_acq: float = 100e-6, n_cycles: int = 200, A0: float = 1, 
        batch_size: int = 1000, noise_level: float = 0.001, apply_filter: bool = True
    ) -> torch.Tensor:
        rf = self.get_RF(n_cycles=n_cycles, A0=A0, t_acq=t_acq, batch_size=batch_size)
        
        if noise_level > 0:
            noise = noise_level * torch.randn_like(rf) * rf.max()
            rf = rf + noise
            
        if apply_filter:
            rf = self.apply_probe_filter(rf)
            
        return rf