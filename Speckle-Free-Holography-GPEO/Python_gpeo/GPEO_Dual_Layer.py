"""
Dual-Layer Metasurface Holography via Gradient-based Progressive-Efficiency Optimization (GPEO)
Reference: "Unlocking Speckle-Free and Near-Unity-Efficiency Holography via a Dual-Layer Metasurface Architecture"

This script strictly reproduces the algorithm presented in the paper, overcoming the fundamental 
trade-off between diffraction efficiency and image fidelity. It features:
1. A Dual-Layer Metasurface Architecture to provide additional degrees of freedom.
2. The GPEO strategy (3 stages: low-efficiency initialization, progressive ramp-up, and final refinement) 
   to achieve pristine image quality (ultra-low speckle contrast, zero phase singularities) 
   alongside near-unity diffraction efficiency.

Refactored for GitHub Release.
"""

import os
import time
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 1. Parameter Configuration (Dual-Layer)
# ==========================================
class Flags:
    def __init__(self):
        # Unit cell parameters
        self.sample_size = 1
        self.Px = 0.25e-6 / self.sample_size  # Pixel pitch X (250 nm)
        self.Py = 0.25e-6 / self.sample_size  # Pixel pitch Y (250 nm)
        self.Nx = 512 * self.sample_size      # Grid size X
        self.Ny = 512 * self.sample_size      # Grid size Y
        self.ratio = 1.0
        self.flag = 1

        # Optical setup parameters
        self.wavelength = 0.532e-6            # Incident light wavelength (532 nm)
        self.refractive_index = [1.00, 1.0]   # Refractive index of propagation media

        # distance[0] = 600 um: Separation distance (d) between the two metasurface layers
        # distance[1] = 200 um: Propagation distance (z) from the second layer to the image plane
        self.distance = [600e-6, 200e-6]      
        self.function = 'gRS'                 # Generalized Rayleigh-Sommerfeld diffraction
        self.NA = None

        # GPEO Training parameters
        self.train_step = 25000               # Total iterations for the 3-stage GPEO process
        self.optim = 'Adam'                   # Optimizer
        self.lr = 3e-3                        # Learning rate
        self.lr_decay_rate = 0.8
        self.stop_threshold = 1e-6

        # Target image directory and filename
        self.data_dir = 'figure/'
        self.filename = 'binary_image.bmp'

        # Loss function and GPEO scheduling parameters
        self.loss_type = 1
        self.decay_effi = 1

        # effi_start corresponds to \eta_{ps} (e.g., 30%) in the paper.
        # This is the initial preset efficiency to establish a singularity-free state.
        self.effi_start = 0.3                 
        self.w_eff = 1.0
        self.w_rmse = 1
        self.w_sd = 1

        # Epochs to save snapshots for Fig. 3 visualization
        self.target_epoch = [10000, 13000, 16000, 18000, 20000, 24900]

        # Generate spatial coordinate grids
        size_x = self.Nx * self.Px
        size_y = self.Ny * self.Py
        self.x = np.linspace(-size_x / 2, size_x / 2, self.Nx + 1)
        self.y = np.linspace(-size_y / 2, size_y / 2, self.Ny + 1)
        self.x_ob = self.ratio * self.x
        self.y_ob = self.ratio * self.y

# ==========================================
# 2. Core Optoelectronic Modeling & Propagation
# ==========================================
class Core:
    def __init__(self, flags):
        self.flags = flags
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cached_target = self.read_figure()

    def build_tensor(self, data, requires_grad=False):
        if isinstance(data, torch.Tensor):
            tensor = data.clone().detach().to(self.device, dtype=torch.float)
        else:
            tensor = torch.tensor(data, device=self.device, dtype=torch.float)
        tensor.requires_grad_(requires_grad)
        return tensor

    def make_optimizer_eval(self, optimizer_type=None):
        optimizer_type = optimizer_type or self.flags.optim
        if optimizer_type == 'Adam':
            return torch.optim.Adam([self.MS_property], lr=self.flags.lr)
        raise ValueError("Optimizer mismatch. Only 'Adam' is supported.")

    @staticmethod
    def piecewise_linear(x, lower_bound, upper_bound, a, b):
        """
        Defines the progressive efficiency ramp-up for the GPEO strategy.
        Stage 1 (x <= lower_bound): Maintain low efficiency 'a' (\eta_{ps}) to eliminate singularities.
        Stage 2 (lower_bound < x < upper_bound): Progressively ramp up efficiency from 'a' to 'b' (\eta_{pf}).
        Stage 3 (x >= upper_bound): Final refinement at near-unity efficiency 'b'.
        """
        if x <= lower_bound:
            return a
        if x >= upper_bound:
            return b
        return a + (b - a) * (x - lower_bound) / (upper_bound - lower_bound)

    @staticmethod
    def gRS(x, y, z, k):
        """Generalized Rayleigh-Sommerfeld (gRS) diffraction kernel."""
        X, Y = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(X ** 2 + Y ** 2 + z ** 2)
        return torch.exp(1j * k * r) * (z / r ** 3) / (2 * np.pi) * (1 - 1j * k * r)

    def Convolution_rs(self, src, x_ap, y_ap, x_ob, y_ob, zp, wavelength, refractive_index):
        """Rigorous wave propagation utilizing Fast Fourier Transform (FFT)."""
        if src.ndim == 2:
            src = src[None, :, :]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)

        refractive_index = self.build_tensor(refractive_index)
        if refractive_index.dim() > 0:
            refractive_index = refractive_index[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(refractive_index.size(0), 1, 1)

        k = 2 * np.pi / wavelength * refractive_index
        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)
        x_ob = self.build_tensor(x_ob)
        y_ob = self.build_tensor(y_ob)

        apsizex = x_ap[-1] - x_ap[0]
        apsizey = y_ap[-1] - y_ap[0]
        obsizex = x_ob[-1] - x_ob[0]
        obsizey = y_ob[-1] - y_ob[0]

        ratio_x = obsizex / apsizex
        ratio_y = obsizey / apsizey

        # Avoid int8 overflow by using native Python int for scaling operations
        ratio_x_int = int(torch.ceil(ratio_x).item())
        ratio_y_int = int(torch.ceil(ratio_y).item())

        Nx1, Ny1 = src.size(-2), src.size(-1)
        Num_x = int(ratio_x.item() * Nx1) if isinstance(ratio_x, torch.Tensor) else int(ratio_x * Nx1)
        Num_y = int(ratio_y.item() * Ny1) if isinstance(ratio_y, torch.Tensor) else int(ratio_y * Ny1)

        sj = torch.linspace(x_ap[0], x_ap[-1], Nx1 + 1, device=self.device)[:-1]
        dsj = sj[1] - sj[0]
        nj = torch.linspace(y_ap[0], y_ap[-1], Ny1 + 1, device=self.device)[:-1]
        dnj = nj[1] - nj[0]

        eval_batchsize = src.size(-3)
        I = torch.zeros(eval_batchsize, ratio_x_int * Nx1, ratio_y_int * Ny1, dtype=torch.complex64, device=self.device)
        U = torch.zeros(eval_batchsize, 2 * Nx1 - 1, 2 * Ny1 - 1, dtype=torch.complex64, device=self.device)
        U[:, 0:Nx1, 0:Ny1] = src

        U_ = fft2(U, dim=(-1, -2))

        for nx in range(ratio_x_int):
            for ny in range(ratio_y_int):
                xj = torch.linspace(x_ob[0] + apsizex * nx, x_ob[0] + apsizex * (nx + 1), Nx1 + 1, device=self.device)[:-1]
                yj = torch.linspace(y_ob[0] + apsizey * ny, y_ob[0] + apsizey * (ny + 1), Ny1 + 1, device=self.device)[:-1]

                Xj = torch.linspace(xj[0] - sj[-1], xj[-1] - sj[0], 2 * Nx1 - 1, device=self.device)
                Yj = torch.linspace(yj[0] - nj[-1], yj[-1] - nj[0], 2 * Ny1 - 1, device=self.device)

                H = Core.gRS(Xj, Yj, zp, k)
                S = ifft2(U_ * fft2(H)) * dsj * dnj
                I[:, nx * Nx1:(nx + 1) * Nx1, ny * Ny1:(ny + 1) * Ny1] = S[:, Nx1 - 1:2 * Nx1 - 1, Ny1 - 1:2 * Ny1 - 1]

        I = I[:, 0:Num_x, 0:Num_y]
        return torch.squeeze(I) if I.size(0) == 1 else I

    def read_figure(self):
        """Read and preprocess the target ground truth image."""
        data_dir = self.flags.data_dir
        apsizex = self.flags.x[-1] - self.flags.x[0]
        apsizey = self.flags.y[-1] - self.flags.y[0]
        obsizex = self.flags.x_ob[-1] - self.flags.x_ob[0]
        obsizey = self.flags.y_ob[-1] - self.flags.y_ob[0]

        ratio_x = obsizex / apsizex
        ratio_y = obsizey / apsizey

        Nx_ob = int(self.flags.Nx * ratio_x)
        Ny_ob = int(self.flags.Ny * ratio_y)

        ratio = 1 / self.flags.ratio
        x1 = int((1 - ratio) / 2 * Nx_ob)
        y1 = int((1 - ratio) / 2 * Ny_ob)

        nxx = int(Nx_ob * ratio)
        nyy = int(Ny_ob * ratio)

        filepath = os.path.join(data_dir, self.flags.filename)
        if not os.path.exists(filepath):
            filepath = self.flags.filename  # Fallback to current directory

        image = cv2.imread(filepath)
        if image is None:
            # Generate a simple mock target image to prevent crashing if missing
            image = np.ones((nyy, nxx, 3), dtype=np.uint8) * 255

        image = image[:, :, 0]
        image = np.rot90(image, -1)

        image = cv2.resize(image, (nyy, nxx))
        max_value = np.max(image) if np.max(image) > 0 else 1
        image = image / max_value

        image0 = np.zeros([Nx_ob, Ny_ob])
        image0[x1:x1 + nxx, y1:y1 + nyy] = image

        return self.build_tensor(image0)

    def initialize_metasurface(self):
        """
        Crucially, as stated in the paper, the initial phase distributions for both metasurface layers 
        are set to be uniform (ones). Random phase initialization leads to high speckle noise and 
        deterministic phase singularities.
        """
        np.random.seed(10)
        Number_ms = len(self.flags.refractive_index)
        sample_size = self.flags.sample_size
        self.MS_property = torch.full([Number_ms, int(self.flags.Nx / sample_size), int(self.flags.Ny / sample_size)], 
                                      1.0, requires_grad=True, device=self.device)

    def model(self, jj=0):
        """
        Dual-layer sequential propagation model:
        1. Light passes through Layer 1 (arphi_1).
        2. Propagates over distance d (e.g., 600 um) to Layer 2.
        3. Light passes through Layer 2 (arphi_2).
        4. Propagates over distance z (e.g., 200 um) to the image plane.
        """
        phase_target1 = torch.repeat_interleave(self.MS_property, self.flags.sample_size, dim=1)
        phase_target2 = torch.repeat_interleave(phase_target1, self.flags.sample_size, dim=2)
        ms_value = torch.exp(1j * 2 * np.pi * phase_target2)

        if ms_value.dim() == 2:
            ms_value = ms_value[None, :, :]

        Num_layer = ms_value.size(0)
        Field = 1
        for i in range(Num_layer):
            MS = ms_value[i]
            Field_pass = MS * Field
            # Propagate field iteratively using ASM/gRS
            Field = self.Convolution_rs(Field_pass, x_ap=self.flags.x, y_ap=self.flags.y, 
                                        x_ob=self.flags.x_ob, y_ob=self.flags.y_ob, 
                                        zp=self.flags.distance[i], 
                                        refractive_index=self.flags.refractive_index[i], 
                                        wavelength=self.flags.wavelength)

            if i == 0:
                self.first_layer = Field
            if i < Num_layer - 1:
                # Center crop field if necessary between layers
                MS_h, MS_w = MS.size()
                Field_h, Field_w = Field.size()

                start_x = (Field_w - MS_w) // 2
                end_x = start_x + MS_w
                start_y = (Field_h - MS_h) // 2
                end_y = start_y + MS_h

                Field = Field[start_y:end_y, start_x:end_x]

        return torch.squeeze(Field)

    def make_loss(self, logit=None, labels=None, iter=0):
        """
        Calculates the MSE loss modified by the preset efficiency parameter \eta_p.
        This function dynamically scales the target intensity based on the current GPEO stage.
        """
        Input = self.MS_property.size(-1) * self.MS_property.size(-2)
        eta = self.flags.w_eff / (torch.sum(labels) / Input)

        mseloss = nn.MSELoss()
        effi_start = getattr(self.flags, 'effi_start', 0.2)

        # Schedule the preset efficiency \eta_p according to GPEO timeline:
        # Stage 1: iterations 0 -> 10000 (constant at effi_start, e.g. 0.3)
        # Stage 2: iterations 10000 -> (train_step - 5000) (ramp-up from effi_start to 1.0)
        b = Core.piecewise_linear(iter, 10000, self.flags.train_step - 5000, effi_start / self.flags.w_eff, 1)

        ratio = b if self.flags.decay_effi == 1 else 1
        # Loss function corresponds to L = || H - (\eta_p * ar{T} / T_{avg}) T ||^2
        mse = mseloss(logit, eta * ratio * labels)
        MSE_loss = 100 * mse

        max_int = torch.max(eta * labels)
        self.mse_256 = torch.sqrt(mse) * 256 / max_int

        # Calculate practical diffraction efficiency \eta_d at region of interest
        index = labels > 0.02
        I_m = logit[index]
        self.image_effi = torch.sum(I_m) / Input

        return MSE_loss

    def speckle_contrast(self, logit):
        """
        Quantifies stochastic speckle via Speckle Contrast (SC).
        Calculated as sqrt(<I^2> - <I>^2) / <I> over the target region.
        """
        Input_image = self.cached_target
        choose_target = Input_image > 0.9
        I = logit[choose_target]
        I_avg = torch.mean(I)
        sc = torch.sqrt(torch.mean(I ** 2) - I_avg ** 2) / I_avg
        return sc

    def train(self):
        """Execution of the Gradient-based Progressive-Efficiency Optimization (GPEO)."""
        self.initialize_metasurface()
        self.optm_eval = self.make_optimizer_eval()
        labels = self.read_figure()

        E2, effi, sc = [], [], []
        self.preset_effi_record = [] # Track \eta_p over time

        print(f"Start Dual-Layer optimization. Total epochs: {self.flags.train_step} ...")
        start_time = time.time()

        for epoch in range(self.flags.train_step):
            self.optm_eval.zero_grad()  
            logit = self.model(epoch)   # Forward calculation to obtain complex field H
            logit = torch.abs(logit) ** 2 # Retrieve intensity |H|^2

            loss = self.make_loss(logit, labels, iter=epoch) 
            loss.backward()  # Backpropagation of the loss gradient
            self.optm_eval.step()  # Update metasurface phases

            # Record preset_efficiency (\eta_p) for plotting Fig 3.
            effi_start = getattr(self.flags, 'effi_start', 0.2)
            b = Core.piecewise_linear(epoch, 10000, self.flags.train_step - 5000, effi_start / self.flags.w_eff, 1)
            self.preset_effi_record.append(b if self.flags.decay_effi == 1 else 1)

            # Record metrics
            sc.append(self.speckle_contrast(logit).detach().clone())
            try:
                effi.append(self.image_effi.detach().clone())
                if epoch in self.flags.target_epoch:
                    E2.append(torch.squeeze(logit).clone().detach())
            except Exception:
                pass

            if epoch % 30 == 0:
                print("Epoch {:4d}, Training loss {:.5f}, Diff. Efficiency (\eta_d) {:.5f}, "
                      "MSE_256 {:.5f}, SC {:.5f}".format(epoch, loss.item(), self.image_effi.item(), 
                                                         self.mse_256.item(), sc[-1].item()))

        print(f"\nOptimization completed in {(time.time() - start_time) / 60:.2f} minutes.")

        self.sc = torch.stack(sc)
        try:
            self.epoch_E2 = torch.stack(E2)
            self.epoch_effi = torch.stack(effi)
        except Exception:
            pass

# ==========================================
# 3. Execution and Chart Generation
# ==========================================
if __name__ == "__main__":
    flags = Flags()
    core_model = Core(flags)
    core_model.train()

    # Extract GPU data to NumPy
    data_E2 = core_model.epoch_E2.cpu().data.numpy()
    data_effi = core_model.epoch_effi.cpu().data.numpy()
    sc = core_model.sc.cpu().data.numpy()

    preset_effi = np.array(core_model.preset_effi_record)
    epoch_list = np.arange(flags.train_step)

    effi_actual = data_effi[flags.target_epoch]
    sc_target = sc[flags.target_epoch]

    print("\n======= Extracted Snapshots for Dual-Layer GPEO =======")
    for i, ep in enumerate(flags.target_epoch):
        print(f"Epoch {ep:5d}: Diffraction Efficiency (\eta_d) = {effi_actual[i]:.4f}, Speckle Contrast (SC) = {sc_target[i]:.4f}")

    # ================= Plotting (Reproducing Fig 3e & Fig 3f) =================
    fig = plt.figure(figsize=(10, 6))
    backcolor = (0.85, 0.85, 0.85)
    cmap_color = 'viridis'

    # 1. Plot evolution curves (Corresponds to Fig 3e)
    ax1 = fig.add_axes([0.1, 0.65, 0.8, 0.3])
    ax1.plot(epoch_list, preset_effi, color='#ed008c', label=r'Preset $\eta_{\mathrm{p}}$', linewidth=2)
    ax1.plot(epoch_list, data_effi, color='#f89217', label=r'Actual $\eta_{\mathrm{d}}$', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Efficiency', color='black', fontsize=12)
    ax1.set_ylim([0.2, 1.05])
    ax1.set_xlim([0, 25000])
    ax1.set_xticks([0, 10000, 20000, 25000])
    ax1.set_yticks([0.3, 0.6, 1.0])
    ax1.set_title('Dual Layer (DL) Evolution during GPEO', fontsize=14, fontweight='bold')

    # Twin axis for Speckle Contrast
    ax2 = ax1.twinx()
    ax2.plot(epoch_list, sc, color='#33b1e5', label='Speckle Contrast (SC)', linewidth=2)
    ax2.set_ylabel('SC', color='black', fontsize=12)
    ax2.set_ylim([0.007, 0.017])

    # Overlay star markers to denote stages
    cmap_rainbow = plt.get_cmap('gist_rainbow')
    shared_colors = cmap_rainbow(np.linspace(0, 0.94, len(flags.target_epoch)))
    ax2.scatter(flags.target_epoch, sc_target, c=shared_colors, marker='*', s=150, zorder=10)

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')

    # 2. Plot images for each evolution stage (Corresponds to Fig 3f)
    width_x1, gap_x1 = 0.13, 0.015
    y_pos, height_y, left_start = 0.2, 0.35, 0.05

    for i in range(len(flags.target_epoch)):
        ax = fig.add_axes([left_start + i * (width_x1 + gap_x1), y_pos, width_x1, height_y])
        cax = ax.imshow(np.rot90(np.abs(data_E2[i])), cmap=cmap_color)
        ax.axis('off')

        # Add \eta_d text box at the bottom
        ax.text(0.05, 0.10, rf'$\eta_{{\mathrm{{d}}}}$: {effi_actual[i]:.2f}', color='red',
                bbox=dict(facecolor=backcolor, alpha=0.95, edgecolor='none', boxstyle='square,pad=0.15'),
                fontsize=10, ha='left', va='bottom', transform=ax.transAxes)

        # Add SC text box at the top
        ax.text(0.05, 0.95, f'SC: {sc_target[i]:.3f}', color='red',
                bbox=dict(facecolor=backcolor, alpha=0.95, edgecolor='none', boxstyle='square,pad=0.15'),
                fontsize=10, ha='left', va='top', transform=ax.transAxes)

        # Add rainbow border to match markers
        height_p, width_p = data_E2[i].shape
        rect = plt.Rectangle((-0.5, -0.5), width_p, height_p, linewidth=4, edgecolor=shared_colors[i], facecolor='none')
        ax.add_patch(rect)

        # Add Colorbar to the rightmost image
        if i == len(flags.target_epoch) - 1:
            cbar_ax = fig.add_axes([left_start + len(flags.target_epoch) * (width_x1 + gap_x1) - 0.005, y_pos, 0.015, height_y])
            cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
            cbar.set_ticks([])

    plt.savefig('GPEO_Double_Layer_Fig3.svg', format='svg', dpi=600)
    plt.savefig('GPEO_Double_Layer_Fig3.png', dpi=300)
    print("\nFigure successfully saved as 'GPEO_Double_Layer_Fig3.svg' and '.png'.")
    plt.show()
