import pytest
import torch
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.simulation.simulation import StableSimulator, BBagSimulator
from pycav.solvers.preproc import compute_csm
from pycav.solvers.beamformers import DAS, RCB, RCB_Li, FB

# ---------------------------------------------------------
# FIXTURES : Objets partagés créés avant chaque test
# ---------------------------------------------------------
@pytest.fixture
def device():
    # On teste sur CPU pour garantir que les tests passent sur n'importe quelle machine
    return "cpu"

@pytest.fixture
def probe(device):
    # Sonde miniature pour des tests rapides (32 éléments)
    return Probe.from_linear(n_el=32, pitch=0.3e-3, fs=20e6, device=device)

@pytest.fixture
def grid(device):
    # Grille minuscule (10x10 pixels = 100 positions)
    return Grid.from_limits(x_lim=(-0.005, 0.005), z_lim=(0.01, 0.02), step=0.001, device=device)

@pytest.fixture
def dummy_rf(probe):
    # Faux signal RF de 1000 samples
    n_samples = 1000
    return torch.randn(n_samples, probe.n_elements, dtype=torch.float32, device=probe.device)

@pytest.fixture
def dummy_csm(probe):
    # Fausse CSM théorique (Produit extérieur d'un vecteur aléatoire)
    # Shape attendue par les solvers: [F, Ne, Ne] -> On prend F=1
    x = torch.randn(probe.n_elements, dtype=torch.complex64, device=probe.device)
    csm = torch.outer(x, x.conj()).unsqueeze(0) 
    return csm

# ---------------------------------------------------------
# 1. TESTS CORE
# ---------------------------------------------------------
def test_probe_creation(probe):
    assert probe.n_elements == 32
    assert probe.positions.shape == (32, 3)
    assert probe.positions.dtype == torch.float32
    assert probe.fc.dtype == torch.float32

def test_grid_creation(grid):
    # 10x10 pixels (environ, selon le step), l'important est la dimension finale
    n_pixels = grid.n_pixels
    assert grid.positions.shape == (n_pixels, 3)
    assert grid.positions.dtype == torch.float32

# ---------------------------------------------------------
# 2. TESTS SIMULATION
# ---------------------------------------------------------
def test_bbag_simulator(probe, device):
    bubble_pos = torch.tensor([[0.0, 0.0, 0.015]], dtype=torch.float32, device=device)
    sim = BBagSimulator(probe, bubble_pos, fhifu=1e6, device=device)
    
    rf = sim.simulate(t_acq=50e-6, n_cycles=10, apply_filter=True)
    
    n_samples_expected = int(50e-6 * probe.fs)
    assert rf.shape == (n_samples_expected, probe.n_elements)
    assert rf.dtype == torch.float32

def test_stable_simulator(probe, device):
    bubble_pos = torch.tensor([[0.0, 0.0, 0.015]], dtype=torch.float32, device=device)
    sim = StableSimulator(probe, bubble_pos, fhifu=1e6, device=device)
    
    rf = sim.simulate(t_acq=50e-6, n_cycles=10, apply_filter=True)
    
    n_samples_expected = int(50e-6 * probe.fs)
    assert rf.shape == (n_samples_expected, probe.n_elements)
    assert rf.dtype == torch.float32

# ---------------------------------------------------------
# 3. TESTS PREPROC (CSM)
# ---------------------------------------------------------
def test_compute_csm(dummy_rf, probe):
    f_target = 5e6
    csm, axf = compute_csm(dummy_rf, probe, K=5, overlap=0.5, f=[f_target])
    
    # Doit renvoyer [F=1, Ne=32, Ne=32]
    assert csm.shape == (1, probe.n_elements, probe.n_elements)
    assert csm.dtype == torch.complex64
    assert axf.dtype == torch.float32

# ---------------------------------------------------------
# 4. TESTS SOLVERS
# ---------------------------------------------------------
@pytest.mark.parametrize("SolverClass", [DAS, RCB, RCB_Li, FB])
def test_solvers(SolverClass, probe, grid, dummy_csm):
    solver = SolverClass(probe, grid, c=1540.0)
    
    freqs = [5e6]
    img = solver.solve(dummy_csm, freqs)
    
    # L'image finale (moyennée sur les fréquences) doit avoir la taille du nombre de pixels
    assert img.shape == grid.shape
    assert img.dtype == torch.float32
    
    # Vérifie qu'aucune instabilité mathématique n'a produit de NaN ou d'Inf
    assert not torch.isnan(img).any(), f"NaN detected in {SolverClass.__name__} output"
    assert not torch.isinf(img).any(), f"Inf detected in {SolverClass.__name__} output"