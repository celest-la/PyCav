# PyCav: High-Performance Passive Cavitation Imaging in PyTorch

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**PyCav** is a modern, GPU-accelerated Python library dedicated to Passive Cavitation Imaging (PCI) and Passive Acoustic Mapping (PAM). By leveraging PyTorch's tensor operations and autograd engine capabilities (while strictly managing VRAM), PyCav transitions traditional, slow CPU-based acoustic mapping (e.g., MATLAB scripts) into highly parallelized, lightning-fast CUDA workflows.

## ✨ Key Features


* **🫧 Cavitation Simulation**: Built-in simulators for both **Broadband (BBag)** and **Stable/Subharmonic** cavitation signatures.
* **🧠 Robust Beamformers**:
  * **DAS** (Delay-and-Sum)
  * **RCB** (Robust Capon Beamformer / Minimum Variance)
  * **RCB_Li** (Advanced RCB with uncertainty constraints)
  * **FB / PISA** (Functional Beamforming)
* **🧮 Fast processing**: GPU friendly architecture

---

## 📦 Installation

It is recommended to use a virtual environment (`venv`, `conda`, or `uv`).

```bash
git clone https://github.com/celest-la/PyCav.git
cd PyCav
pip install -e .
```
*(Dependencies will be installed automatically).*

---

##  Quickstart

Here is a complete, end-to-end example: from defining the geometry and simulating a stable cavitation bubble, to reconstructing the acoustic map using the Robust Capon Beamformer (RCB).

```python
import torch
import matplotlib.pyplot as plt
from pycav.core.probe import Probe
from pycav.core.grid import Grid
from pycav.simulation.simulation import StableSimulator
from pycav.solvers.preproc import compute_csm
from pycav.solvers.beamformers import RCB
from pycav.utils.io import plot_result_dB

# 1. Setup Environment & Geometry
device = "cuda" if torch.cuda.is_available() else "cpu"

probe = Probe.from_linear(n_el=128, pitch=0.3e-3, fs=22e6, device=device)
grid = Grid.from_limits(x_lim=(-0.01, 0.01), z_lim=(0.06, 0.08), step=0.0002, device=device)

# 2. Simulate a Stable Cavitation Event
bubble_pos = torch.tensor([[0.0, 0.0, 0.07]], device=device) # Bubble at z=7cm
sim = StableSimulator(probe, bubble_pos, fhifu=1e6, device=device)

# 1-line simulation: propagation, subharmonic kernel convolution, noise, and L7-5 filtering
rf_data = sim.simulate(t_acq=1e-3, n_cycles=500, noise_level=0.05)

# 3. Pre-Processing (Compute Cross-Spectral Matrix)
# Looking at the 4.5 MHz harmonic
csm, freqs = compute_csm(rf_data, probe, K=100, overlap=0.9, f=[4.5e6])

# 4. Reconstruction (Beamforming)
rcb = RCB(probe, grid, c=1540.0)
img = rcb.solve(csm, freqs, epsilon=10)

# 5. Visualization
plt.figure(figsize=(6, 5))
plot_result_dB(img, grid, title="RCB Reconstruction", dynamic_range=20)
plt.show()
```

---

## 🏗️ Project Architecture

* `pycav.core`: Foundational geometries (`Grid`, `Probe`).
* `pycav.physics`: Acoustic wave propagation and steering vector computations.
* `pycav.simulation`: RF signal generation (`StableSimulator`, `BBagSimulator`).
* `pycav.solvers`: 
  * `preproc`: Cross-Spectral Matrix (`compute_csm`).
  * `beamformers`: State-of-the-art PCI solvers (`DAS`, `RCB`, `FB`, etc.).
* `pycav.utils`: Plotting tools, colormaps (dB mapping), and decorators.

---

## 🧪 Running Tests

PyCav comes with a rigorous test suite validating tensor dimensionalities, `float32`/`complex64` data integrity, and mathematical stability.

To run the tests:
```bash
pip install pytest
pytest
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

***