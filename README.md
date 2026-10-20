# PyCav: High-Performance Passive Cavitation Imaging in PyTorch

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**PyCav** is a modern, GPU-accelerated Python library dedicated to Passive Cavitation Imaging (PCI) and Passive Acoustic Mapping (PAM). By leveraging PyTorch's tensor operations and autograd engine capabilities (while strictly managing VRAM), PyCav transitions traditional, slow CPU-based acoustic mapping (e.g., MATLAB scripts) into highly parallelized, lightning-fast CUDA workflows.

## Key Features


* **Cavitation Simulation**: Built-in simulators for both **Broadband (BBag)** and **Stable/Subharmonic** cavitation signatures.
* **Robust Beamformers**:
  * **DAS** (Delay-and-Sum)
  * **RCB** (Robust Capon Beamformer / Minimum Variance)
  * **FB / PISA** (Functional Beamforming)
* **Fast processing**: GPU friendly architecture

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
There are 2 demos, for 3D et 2D reconstruction.
2D: `demo_reconstruction2D.py`
3D: `demo_reconstruction3D.py`

---

## 🏗️ Project Architecture

* `pycav.core`: Foundational geometries (`Grid`, `Probe`).
* `pycav.physics`: Acoustic wave propagation and steering vector computations.
* `pycav.simulation`: RF signal generation (`StableSimulator`, `BBagSimulator`).
* `pycav.solvers`: 
  * `preproc`: Cross-Spectral Matrix (`compute_csm`).
  * `beamformers`: State-of-the-art PCI solvers (`DAS`, `RCB`, `FB`, etc.).
* `pycav.utils`: Plotting tools, colormaps (dB mapping), and decorators.


## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

***
