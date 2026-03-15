# Dual-Layer Metasurface Holography (GPEO) & Benchmark Framework

This repository provides the complete open-source codebase for the paper: **"Unlocking Speckle-Free and Near-Unity-Efficiency Holography via a Dual-Layer Metasurface Architecture."**

The project is divided into two main components:

## 1. Python GPEO Implementation (`/python_gpeo`)
The Python directory contains the official implementation of the **Gradient-based Progressive-Efficiency Optimization (GPEO)** algorithm. It utilizes PyTorch and rigorous angular spectrum methods to design dual-layer metasurfaces that break the fundamental trade-off between diffraction efficiency and image fidelity.
- **Key Features:** Achieves near-unity efficiency (>95%) with pristine image quality (ultra-low speckle contrast and zero phase singularities).

## 2. MATLAB Benchmark Framework (`/matlab_benchmark`)
The MATLAB directory provides a rigorous simulation framework for evaluating and comparing various classical computer-generated holography (CGH) phase retrieval algorithms under fair physical constraints. 
- **Evaluated Algorithms:** GS (Baseline), DCGS, BCGS, and AWGS.

---
Please navigate to the respective folders for detailed instructions, requirements, and execution scripts for each tool.
