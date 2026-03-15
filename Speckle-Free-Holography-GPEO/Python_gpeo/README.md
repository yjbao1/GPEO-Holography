# main_gpeo

Python implementation of **Gradient-based Progressive-Efficiency Optimization (GPEO)** for holographic image reconstruction with a **dual-layer metasurface architecture**.

This module is part of the codebase for the paper:

**Unlocking Speckle-Free and Near-Unity-Efficiency Holography via a Dual-Layer Metasurface Architecture**

The goal of this implementation is to reproduce the core idea of the paper: breaking the long-standing trade-off between **diffraction efficiency** and **image fidelity** in holography by combining:
- a **dual-layer metasurface design**
- a **gradient-based progressive-efficiency optimization strategy**
- a **uniform phase initialization**
- rigorous wave propagation modeling based on RS diffraction calculation

---

## Overview

Traditional single-layer holographic optimization often faces a zero-sum trade-off: improving diffraction efficiency tends to worsen speckle noise or introduce deterministic dark defects such as phase singularities.

This `main_gpeo` module implements the **GPEO strategy**, which progressively drives the system from a low-efficiency but singularity-free state toward near-unity diffraction efficiency, while preserving high image quality.

In the dual-layer setting, the additional degrees of freedom provided by two cascaded phase planes significantly improve controllability compared with a single-layer design.

---

## Key Idea of GPEO

The optimization follows a **three-stage schedule**:

1. **Stage 1: Low-efficiency initialization**
   - The optimization begins from a low preset efficiency (for example, 30%).
   - This stage is used to reach a stable and singularity-free solution.

2. **Stage 2: Progressive efficiency ramp-up**
   - The preset efficiency is gradually increased toward 100%.
   - This allows the optimization to redirect more optical energy into the target image without abruptly destroying the image quality obtained in Stage 1.

3. **Stage 3: Final refinement**
   - The efficiency target is kept near unity.
   - The model continues optimizing for final convergence and image refinement.

This strategy is motivated by the observation that once the optimization is initialized in a zero-singularity state, the singularity number tends to remain conserved during the later efficiency ramp-up.

---

## What this script does

The Python script in this folder:

- initializes a **dual-layer metasurface phase profile**
- propagates the optical field layer by layer
- reconstructs the holographic image intensity
- computes the loss between the reconstructed image and the target image
- updates the phase distributions using **Adam**
- tracks:
  - **diffraction efficiency**
  - **speckle contrast (SC)**
  - intermediate reconstruction snapshots
- generates plots similar to the evolution figure shown in the manuscript

---

## Physical Setup

The default implementation uses the following simulation setting:

- Wavelength: **532 nm**
- Metasurface size: **512 × 512**
- Pixel pitch: **250 nm**
- Layer spacing: **600 µm**
- Image-plane propagation distance: **200 µm**
- Optimizer: **Adam**
- Learning rate: **0.003**
- Total optimization steps: **25,000**

These settings match the dual-layer GPEO reproduction script included in this folder.

---

## File Structure

A recommended structure is:

```text
Python_gpeo/
├── README.md
├── GPEO_Dual_Layer.py
└── figure/
    └── binary_image.bmp
