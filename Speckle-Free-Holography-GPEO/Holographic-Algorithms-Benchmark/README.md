# Holographic Algorithms Benchmark Framework

This repository provides a rigorous MATLAB-based benchmark framework for simulating, evaluating, and comparing various computer-generated holography (CGH) phase retrieval algorithms. It is designed to objectively evaluate diffraction efficiency and speckle contrast (SC) under fair and strict physical constraints.

## Overview

Traditional single-layer holographic algorithms face an inherent trade-off between diffraction efficiency and speckle contrast. To benchmark this zero-sum game, this script evaluates four classical algorithms:

1. **GS (Baseline)**: Traditional Gerchberg-Saxton algorithm.
2. **DCGS [1]**: Double-Constraint Gerchberg-Saxton algorithm.
3. **BCGS [2]**: Bandwidth Constraint strategy using a pure zero-padded FFT far-field model.
4. **AWGS [3]**: Adaptive Weighted Gerchberg-Saxton algorithm.

> **Note:** Our proposed GPEO algorithm completely breaks this physical trade-off. This benchmark script serves as the comparative baseline demonstrating the limitations of conventional single-layer methods.
> - **MAPS limit [4]**: Efficiency = 29.6%, SC = 0.079
> - **Our Work (GPEO)**: **Efficiency = 97.0%, SC = 0.013**

## Key Features

- **Accurate Physical Propagation**: GS, DCGS, and AWGS strictly utilize the Rayleigh-Sommerfeld (RS) propagation model (`RS_FFT_s`) to perfectly match near-field/mid-field physical diffraction.
- **Energy Conservation**: BCGS is implemented via a pure zero-padded FFT model with strict unitary energy conservation to perfectly reproduce the original paper's band-limited methodology.
- **Dual-Mask Evaluation**: 
  - `iter_mask`: Constraints are applied uniformly across the central target domain during iterations.
  - `eval_mask`: Evaluation metrics (Efficiency & SC) are calculated *strictly* over the high-intensity target pattern for absolute objectivity.

## How to Run

1. Clone or download this repository to your local machine.
2. Ensure both `main_benchmark.m` and `binary_image.bmp` are in the same directory.
3. Open `main_benchmark.m` in MATLAB.
4. Run the script. The console will output the metrics, and a figure window will display the final reconstructed center intensity maps for all four algorithms side-by-side.

## Requirements
- MATLAB (Tested on R2021a and newer versions)
- Image Processing Toolbox

## References

1. Chang, C., Xia, J., Yang, L., Lei, W., Yang, Z., & Chen, J. "Speckle-suppressed phase-only holographic three-dimensional display based on double-constraint Gerchberg-Saxton algorithm." *Applied Optics*, 54(23), 6994-7001 (2015).
2. Chen, L., Tian, S., Zhang, H., Cao, L., & Jin, G. "Phase hologram optimization with bandwidth constraint strategy for speckle-free optical reconstruction." *Optics Express*, 29(8), 11645-11661 (2021).
3. Wu, Y., Wang, J., Chen, C., Liu, C. J., Jin, F. M., & Chen, N. "Adaptive weighted Gerchberg-Saxton algorithm for generation of phase-only hologram with artifacts suppression." *Optics Express*, 29(2), 1412-1427 (2021).
4. Zheng, et al. "Phase-probability shaping for speckle-free holographic lithography." *Nature Communications*, (2025).

---
*This code is released under the MIT License.*
