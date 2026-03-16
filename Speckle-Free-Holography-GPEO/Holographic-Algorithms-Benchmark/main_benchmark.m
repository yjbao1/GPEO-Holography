% =========================================================================
% == EXACT LITERATURE-MATCHED HOLOGRAPHIC ALGORITHM COMPARISONS ==
% =========================================================================
% Feature Description:
% 1. GS, DCGS, AWGS use the rigorous Rayleigh-Sommerfeld (RS) propagation
%    model (RS_FFT_s) to match near-field physical diffraction.
% 2. BCGS [Ref 25] uses a pure fft2 far-field model with energy conservation
%    to perfectly reproduce the original paper's band-limited methodology.
% 3. Three-mask design (target is a binary image; imresize introduces
%    fractional boundary pixels with values between 0 and 1):
%
%    - iter_mask    : The central Nx*Ny physical region in the padded
%                    observation plane. Used to confine the amplitude
%                    constraint to the target window during each iteration.
%
%    - eval_eff_mask: target_amp > 0.0  (all non-zero pixels, including
%                    imresize boundary transitions). Intentionally broad
%                    to capture every signal-carrying pixel, ensuring the
%                    computed Efficiency reflects the true fraction of power
%                    delivered to the pattern without under-counting.
%
%    - eval_sc_mask : target_amp > 0.9  (high-confidence bright pixels only,
%                    well above the imresize-induced 0~1 transition zone).
%                    Excludes blurred boundary pixels so that the Speckle
%                    Contrast (SC) metric is not contaminated by the
%                    artificial intensity gradient at pattern edges.
%
% Theoretical / Published Benchmarks for Comparison:
% - MAPS [Ref 27 (Nat. Commun.)] : Efficiency = 29.60%, SC = 0.079
% - Our Work (GPEO proposed)     : Efficiency = 97.00%, SC = 0.013
% =========================================================================
clear; close all; clc;

% =========================================================================
% 1. Initialize Physical and System Parameters
% =========================================================================
disp('--- Initializing Parameters ---');
params.Px = 0.25e-6;         % Hologram pixel pitch in x-direction (m)
params.Py = 0.25e-6;         % Hologram pixel pitch in y-direction (m)
params.Nx = 512;             % Number of hologram pixels in x
params.Ny = 512;             % Number of hologram pixels in y
params.lambda = 0.532e-6;    % Wavelength (m), green laser
params.zp = 200e-6;          % Propagation distance from hologram to image plane (m)
params.Lx = params.Px * params.Nx;   % Physical size of hologram in x (m)
params.Ly = params.Py * params.Ny;   % Physical size of hologram in y (m)
params.apx = [-params.Lx/2, params.Lx/2];   % Hologram aperture extent in x
params.apy = [-params.Ly/2, params.Ly/2];   % Hologram aperture extent in y
ratio = 2;                   % Observation plane is ratio x larger than aperture plane
params.obx = ratio * params.apx;     % Observation plane extent in x
params.oby = ratio * params.apy;     % Observation plane extent in y
params.Nx_ob = ratio * params.Nx;    % Number of pixels in observation plane, x
params.Ny_ob = ratio * params.Ny;    % Number of pixels in observation plane, y
iterations = 200;            % Number of iterations for each algorithm

% =========================================================================
% 2. Load Target Image and Generate Padded Masks
% =========================================================================
chemin = 'F:\Project\2026_speckle_free_hologram\python4\figure\binary_image.bmp';
disp(['Loading target image from: ', chemin]);
XRGB = imread(chemin);
I_core = imresize(XRGB(:,:,1), [params.Nx, params.Ny]);   % Resize to Nx x Ny
I_core = imrotate(I_core, -90);                            % Rotate -90° to correct orientation
I_core = sqrt(double(I_core));             % Convert intensity to amplitude (sqrt)
A_core = I_core ./ max(max(I_core));       % Normalize amplitude to [0, 1]

% Embed the Nx*Ny target amplitude into the zero-padded Nx_ob*Ny_ob observation grid
target_amp = zeros(params.Nx_ob, params.Ny_ob);
start_x = floor((params.Nx_ob - params.Nx) / 2) + 1;
end_x   = start_x + params.Nx - 1;
start_y = floor((params.Ny_ob - params.Ny) / 2) + 1;
end_y   = start_y + params.Ny - 1;
target_amp(start_x:end_x, start_y:end_y) = A_core;

% [Mask 1] iter_mask: Spatial support for iterative amplitude constraints.
% Covers the central Nx*Ny region inside the zero-padded Nx_ob*Ny_ob grid.
iter_mask = false(params.Nx_ob, params.Ny_ob);
iter_mask(start_x:end_x, start_y:end_y) = true;

% [Mask 2] eval_eff_mask: All pixels where target amplitude > 0.
% Uses a loose threshold to include imresize-induced boundary transition
% pixels (0 < value < 1), so that Efficiency accounts for the full signal
% region without under-counting edge power.
eval_eff_mask = target_amp > 0.0;

% [Mask 3] eval_sc_mask: Only high-confidence bright pixels (amplitude > 0.9).
% Excludes the 0~1 boundary zone introduced by imresize, so that Speckle
% Contrast (SC) is computed purely on the well-defined interior of the
% pattern, free from edge-blurring artifacts.
eval_sc_mask = target_amp > 0.9;

% =========================================================================
% 3. Run Algorithm Benchmarks
% =========================================================================
disp('--- Starting Algorithm Benchmarks ---');
results = struct('Algorithm', {}, 'Efficiency', {}, 'SC', {});

% --- 1. Traditional GS (RS Propagation) ---
[eff, sc, final_phase_gs, final_image_gs] = run_GS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations);
results(end+1) = struct('Algorithm', 'Traditional GS', 'Efficiency', eff, 'SC', sc);

% --- 2. DCGS [Ref 23] (RS Propagation) ---
[eff, sc, final_phase_dcgs, final_image_dcgs] = run_DCGS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations);
results(end+1) = struct('Algorithm', 'DCGS [Ref 22]', 'Efficiency', eff, 'SC', sc);

% --- 3. BCGS [Ref 25] (Pure FFT Model) ---
[eff, sc, final_phase_bcgs, final_image_bcgs] = run_BCGS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations);
results(end+1) = struct('Algorithm', 'BCGS [Ref 24]', 'Efficiency', eff, 'SC', sc);

% --- 4. AWGS [Ref 26] (RS Propagation) ---
[eff, sc, final_phase_awgs, final_image_awgs] = run_AWGS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations);
results(end+1) = struct('Algorithm', 'AWGS [Ref 25]', 'Efficiency', eff, 'SC', sc);

% =========================================================================
% 4. Print Results, Plot in MATLAB, and Save Data for Python
% =========================================================================
disp('--- Benchmarks Complete ---');
results_table = struct2table(results);
disp(results_table);

% Print reference values from literature for direct comparison
fprintf('\n--- Literature & Proposed Method References ---\n');
fprintf('MAPS [Ref 26]      : Efficiency = 29.60%%, SC = 0.079\n');
fprintf('Our Work (GPEO)    : Efficiency = 97.00%%, SC = 0.013\n');
fprintf('-----------------------------------------------\n');

% Crop the reconstructed intensity maps to the central physical region (Nx x Ny).
% Rotate by 90° here so both MATLAB preview and Python export share the same orientation.
img_gs   = rot90(final_image_gs(start_x:end_x, start_y:end_y), 1);
img_dcgs = rot90(final_image_dcgs(start_x:end_x, start_y:end_y), 1);
img_bcgs = rot90(final_image_bcgs(start_x:end_x, start_y:end_y), 1);
img_awgs = rot90(final_image_awgs(start_x:end_x, start_y:end_y), 1);

% Extract metrics from the results table
eff_gs   = results_table.Efficiency(1);  sc_gs   = results_table.SC(1);
eff_dcgs = results_table.Efficiency(2);  sc_dcgs = results_table.SC(2);
eff_bcgs = results_table.Efficiency(3);  sc_bcgs = results_table.SC(3);
eff_awgs = results_table.Efficiency(4);  sc_awgs = results_table.SC(4);

% -------------------------------------------------------------------------
% MATLAB Plotting (2x2 Layout Preview)
% -------------------------------------------------------------------------
figure('Name', 'Final Reconstructed Intensities', 'Position', [100, 100, 800, 700]);

% (a) Traditional GS
subplot(2, 2, 1);
imagesc(img_gs); axis image off;
title(sprintf('(a) Traditional GS\nEff: %.2f%%, SC: %.3f', eff_gs*100, sc_gs), ...
    'FontSize', 12, 'FontWeight', 'bold');
colorbar;

% (b) DCGS [Ref 23]
subplot(2, 2, 2);
imagesc(img_dcgs); axis image off;
title(sprintf('(b) DCGS [Ref 22]\nEff: %.2f%%, SC: %.3f', eff_dcgs*100, sc_dcgs), ...
    'FontSize', 12, 'FontWeight', 'bold');
colorbar;

% (c) BCGS [Ref 25]
subplot(2, 2, 3);
imagesc(img_bcgs); axis image off;
title(sprintf('(c) BCGS [Ref 24]\nEff: %.2f%%, SC: %.3f', eff_bcgs*100, sc_bcgs), ...
    'FontSize', 12, 'FontWeight', 'bold');
colorbar;

% (d) AWGS [Ref 26]
subplot(2, 2, 4);
imagesc(img_awgs); axis image off;
title(sprintf('(d) AWGS [Ref 25]\nEff: %.2f%%, SC: %.3f', eff_awgs*100, sc_awgs), ...
    'FontSize', 12, 'FontWeight', 'bold');
colorbar;

% Apply colormap (Fallback to parula if viridis is not supported in older MATLAB versions)
try
    colormap(viridis);
catch
    warning('Viridis colormap not found. Falling back to parula.');
    colormap(parula);
end
set(gcf, 'Color', 'w'); % Set figure background to white

% -------------------------------------------------------------------------
% Save Data for Python Plotting
% -------------------------------------------------------------------------
save('benchmark_results.mat', 'img_gs', 'img_dcgs', 'img_bcgs', 'img_awgs', ...
     'eff_gs', 'sc_gs', 'eff_dcgs', 'sc_dcgs', 'eff_bcgs', 'sc_bcgs', 'eff_awgs', 'sc_awgs');
disp('>>> MATLAB Plot generated.');
disp('>>> Data successfully saved to "benchmark_results.mat" for high-quality Python plotting.');

% =========================================================================
% == Algorithm Internal Implementations ==
% =========================================================================

function [eff, sc, final_phase, final_image_intensity] = run_GS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations)
    fprintf('\nRunning Traditional GS...\n');
    tic;
    % Initialize hologram with uniform phase (overwritten on next line)
    holo_field = ones(params.Nx, params.Ny);
    holo_field = exp(1i * 2 * pi * ones(params.Nx, params.Ny)); % Phase-only, uniform phase
    for iter = 1:iterations
        % Forward propagation: hologram plane -> image plane (RS diffraction)
        img_field = RS_FFT_s(holo_field, params.apx, params.apy, params.obx, params.oby, ...
            params.zp, params.lambda, 'method', 'RS');
        % Image plane constraint: replace amplitude with target, keep retrieved phase
        img_field_constrained = target_amp .* exp(1i * angle(img_field));
        % Backward propagation: image plane -> hologram plane (inverse RS)
        holo_field_new = RS_FFT_s(img_field_constrained, params.obx, params.oby, params.apx, params.apy, ...
            params.zp, params.lambda, 'method', 'RS_inverse');
        % Hologram plane constraint: phase-only SLM (discard amplitude)
        holo_field = 1.0 .* exp(1i * angle(holo_field_new));
    end
    final_phase = angle(holo_field);
    final_img_field = img_field;   % Use the last forward propagation result
    final_image_intensity = abs(final_img_field).^2;
    [eff, sc] = calculate_metrics(final_image_intensity, eval_sc_mask, eval_eff_mask, params);
    fprintf('GS finished in %.2f seconds.\n', toc);
end

function [eff, sc, final_phase, final_image_intensity] = run_DCGS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations)
    fprintf('\nRunning DCGS [Ref 22]...\n');
    tic;
    % Initialize hologram with random phase
    holo_field = exp(1i * 2 * pi * rand(params.Nx, params.Ny));
    beta = 0.5;   % Double-constraint feedback parameter (controls over-correction strength)
    for iter = 1:iterations
        % Forward propagation: hologram plane -> image plane (RS diffraction)
        img_field = RS_FFT_s(holo_field, params.apx, params.apy, params.obx, params.oby, ...
            params.zp, params.lambda, 'method', 'RS');
        A_n = abs(img_field);        % Current amplitude in image plane
        phi_n = angle(img_field);    % Current phase in image plane
        amp_new   = A_n;             % Initialize updated amplitude with current values
        phase_new = phi_n;           % Phase is kept unchanged everywhere (including signal region)

        % Compute normalization factor alpha: ratio of current to target amplitude sum in signal region
        sum_A_S = sum(A_n(iter_mask));
        sum_T_S = sum(target_amp(iter_mask));
        alpha_val = sum_A_S / max(sum_T_S, eps);

        % Double-constraint amplitude update in signal region: 2*alpha*T - beta*A_n
        % Negative values clipped to zero to maintain physical validity
        sig_amp = 2 * alpha_val * target_amp(iter_mask) - beta * A_n(iter_mask);
        amp_new(iter_mask) = max(sig_amp, 0);
        phase_new(iter_mask) = 0; % Force phase to 0 in target region

        % Apply updated amplitude with unchanged phase, then back-propagate
        img_field_constrained = amp_new .* exp(1i * phase_new);
        holo_field_new = RS_FFT_s(img_field_constrained, params.obx, params.oby, params.apx, params.apy, ...
            params.zp, params.lambda, 'method', 'RS_inverse');
        % Hologram plane constraint: phase-only SLM
        holo_field = 1.0 .* exp(1i * angle(holo_field_new));
    end
    final_phase = angle(holo_field);
    % Final forward propagation for clean intensity readout
    final_img_field = RS_FFT_s(exp(1i * final_phase), params.apx, params.apy, params.obx, params.oby, ...
        params.zp, params.lambda, 'method', 'RS');
    final_image_intensity = abs(final_img_field).^2;
    [eff, sc] = calculate_metrics(final_image_intensity, eval_sc_mask, eval_eff_mask, params);
    fprintf('DCGS finished in %.2f seconds.\n', toc);
end

function [eff, sc, final_phase, final_image_intensity] = run_BCGS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations)
    fprintf('\nRunning BCGS [Ref 24] with Pure FFT...\n');
    tic;
    [Ny_ob, Nx_ob] = size(target_amp);   % MATLAB size(): first dim = rows = Ny, second = cols = Nx
    N_total = Ny_ob * Nx_ob;             % Total pixel count, used as FFT energy normalization factor

    % Build a binary hologram mask: 1 inside the central Nx*Ny aperture, 0 outside
    holo_mask = zeros(Ny_ob, Nx_ob);
    start_y = floor((Ny_ob - params.Ny) / 2) + 1;
    end_y   = start_y + params.Ny - 1;
    start_x = floor((Nx_ob - params.Nx) / 2) + 1;
    end_x   = start_x + params.Nx - 1;
    holo_mask(start_y:end_y, start_x:end_x) = 1;

    % Quadratic phase chirp parameters for band-limiting initialization
    k_param = 2 / params.Nx;   % Chirp rate in x
    l_param = 2 / params.Ny;   % Chirp rate in y
    m_lin = linspace(-Nx_ob/2, Nx_ob/2 - 1, Nx_ob);
    n_lin = linspace(-Ny_ob/2, Ny_ob/2 - 1, Ny_ob);
    [M, N] = meshgrid(m_lin, n_lin);

    % Initial quadratic phase: exp(i*pi*(k*m^2 + l*n^2)), applied as initial image field
    initial_phase = exp(1i * pi * (k_param * M.^2 + l_param * N.^2));
    if ~isequal(size(initial_phase), size(target_amp)), initial_phase = initial_phase.'; end

    % Seed the constrained image field with target amplitude and initial quadratic phase
    img_field_constrained = target_amp .* initial_phase;

    % Energy-conserving (unitary) FFT operators
    forward_prop  = @(u) fftshift(fft2(ifftshift(u))) / sqrt(N_total);
    backward_prop = @(u) fftshift(ifft2(ifftshift(u))) * sqrt(N_total);

    for iter = 1:iterations
        % Backward propagation: image plane -> hologram plane
        holo_field_full = backward_prop(img_field_constrained);
        % Hologram plane constraint: phase-only within aperture mask, zero outside
        holo_field_constrained = holo_mask .* exp(1i * angle(holo_field_full));

        % Forward propagation: hologram plane -> image plane
        img_field = forward_prop(holo_field_constrained);
        amp_new = abs(img_field);
        % Image plane constraint: replace amplitude in signal region with target amplitude
        amp_new(iter_mask) = target_amp(iter_mask);
        img_field_constrained = amp_new .* exp(1i * angle(img_field));
    end

    % Final readout
    final_img_field = forward_prop(holo_field_constrained);
    final_image_intensity = abs(final_img_field).^2;
    final_phase_full = angle(holo_field_constrained);
    % Extract phase from the central Nx*Ny aperture region only
    final_phase = final_phase_full(start_y:end_y, start_x:end_x);
    [eff, sc] = calculate_metrics(final_image_intensity, eval_sc_mask, eval_eff_mask, params);
    fprintf('BCGS finished in %.2f seconds.\n', toc);
end

function [eff, sc, final_phase, final_image_intensity] = run_AWGS(target_amp, iter_mask, eval_sc_mask, eval_eff_mask, params, iterations)
    fprintf('\nRunning AWGS [Ref 25]...\n');
    tic;

    % --- Ref 25 Eq(3): Quadratic Phase Initialization ---
    % Generating normalized pixel coordinates (p, q) in [-1, 1]
    p = linspace(-1, 1, params.Nx);
    q = linspace(-1, 1, params.Ny);
    [P, Q] = meshgrid(p, q);

    % Chirp parameters a and b, both in (0, 1)
    a_param = 0.5;
    b_param = 0.5;

    % Initial hologram field: quadratic phase exp(i*pi*(a*p^2 + b*q^2))
    initial_phase = exp(1i * pi * (a_param * P.^2 + b_param * Q.^2));
    holo_field = initial_phase;

    % --- Iterative Process ---
    for iter = 1:iterations
        % Forward propagation: hologram plane -> image plane (RS diffraction)
        img_field = RS_FFT_s(holo_field, params.apx, params.apy, params.obx, params.oby, ...
            params.zp, params.lambda, 'method', 'RS');
        amp_img = abs(img_field);
        amp_new = amp_img;   % Initialize updated amplitude with current values

        % Ref 25 adaptive weight: w = exp(A_t - A_r)
        % Amplifies under-reconstructed pixels and suppresses over-reconstructed ones
        A_t = target_amp(iter_mask);   % Target amplitude in signal region
        A_r = amp_img(iter_mask);      % Current reconstructed amplitude in signal region
        w_pro = exp(A_t - A_r);        % Pixel-wise adaptive weight

        % Apply weighted amplitude constraint in signal region
        amp_new(iter_mask) = A_t .* w_pro;

        % Backward propagation: constrained image field -> hologram plane (inverse RS)
        img_field_constrained = amp_new .* exp(1i * angle(img_field));
        holo_field_new = RS_FFT_s(img_field_constrained, params.obx, params.oby, params.apx, params.apy, ...
            params.zp, params.lambda, 'method', 'RS_inverse');

        % Hologram plane constraint: phase-only SLM
        holo_field = 1.0 .* exp(1i * angle(holo_field_new));
    end

    % Final readout
    final_phase = angle(holo_field);
    final_img_field = RS_FFT_s(exp(1i * final_phase), params.apx, params.apy, params.obx, params.oby, ...
        params.zp, params.lambda, 'method', 'RS');
    final_image_intensity = abs(final_img_field).^2;

    [eff, sc] = calculate_metrics(final_image_intensity, eval_sc_mask, eval_eff_mask, params);
    fprintf('AWGS finished in %.2f seconds.\n', toc);
end

function [eff, sc] = calculate_metrics(image_intensity, eval_sc_mask, eval_eff_mask, params)
    % Efficiency: fraction of total hologram-plane power delivered to the signal region.
    % eval_eff_mask uses a loose threshold (> 0.0) to avoid under-counting edge pixels
    % caused by imresize boundary interpolation.
    signal_eff_power = sum(image_intensity(eval_eff_mask));
    total_power = params.Nx * params.Ny;   % Reference power = number of hologram pixels (unit amplitude input)
    eff = signal_eff_power / total_power;

    % Speckle Contrast (SC) = std / mean, computed only on high-confidence bright pixels.
    % eval_sc_mask uses a strict threshold (> 0.9) to exclude imresize-induced 0~1
    % boundary transition pixels, preventing edge artifacts from inflating SC.
    signal_pixels = image_intensity(eval_sc_mask);
    sc = std(signal_pixels) / mean(signal_pixels);
end


% =========================================================================
% Algorithm adapted from:
% "Fast-Fourier-transform based numerical integration method for the
%  Rayleigh-Sommerfeld diffraction formula" by Fabin Shen and Anbo Wang.
function [I] = RS_FFT_s(src, x_ap, y_ap, x_ob, y_ob, z_p, lambda, varargin)
    % Parse optional 'method' argument: 'RS' (forward) or 'RS_inverse' (backward)
    method = 'RS';
    for i = 1:2:length(varargin)
        if strcmp(varargin{i}, 'method')
            method = varargin{i+1};
        end
    end

    k = 2 * pi / lambda;              % Wave number
    apsizex = max(x_ap) - min(x_ap);  % Aperture (hologram) physical width in x
    apsizey = max(y_ap) - min(y_ap);  % Aperture (hologram) physical width in y
    obsizex = max(x_ob) - min(x_ob);  % Observation plane physical width in x
    obsizey = max(y_ob) - min(y_ob);  % Observation plane physical width in y

    [Nx, Ny] = size(src);

    % Tile observation plane into windows matching the aperture size for sub-region RS integration
    index_x = obsizex / apsizex; Nobsx = ceil(index_x);  % Number of tiles in x
    index_y = obsizey / apsizey; Nobsy = ceil(index_y);  % Number of tiles in y

    I = zeros(Nobsx * Nx, Nobsy * Ny);   % Pre-allocate full output field

    % Source plane sampling coordinates
    sj = linspace(min(x_ap), max(x_ap), Nx + 1); sj = sj(1:end-1);
    nj = linspace(min(y_ap), max(y_ap), Ny + 1); nj = nj(1:end-1);

    % Zero-pad source field to (2Nx-1) x (2Ny-1) for linear (non-circular) convolution via FFT
    U = zeros(2 * Nx - 1, 2 * Ny - 1);
    U(1:Nx, 1:Ny) = src;

    U_ = fft2(U);         % Pre-compute FFT of padded source
    dsj = sj(2) - sj(1);  % Sampling interval in x
    dnj = nj(2) - nj(1);  % Sampling interval in y

    % Loop over each observation sub-window tile
    for nx = 0 : Nobsx-1
        for ny = 0 : Nobsy-1
            % Observation coordinates for current tile
            xj = linspace(min(x_ob) + nx*apsizex, min(x_ob) + (nx+1)*apsizex, Nx+1); xj = xj(1:end-1);
            yj = linspace(min(y_ob) + ny*apsizey, min(y_ob) + (ny+1)*apsizey, Ny+1); yj = yj(1:end-1);

            % Difference coordinate vectors for the RS Green's function kernel
            Xj = linspace(min(xj)-max(sj), max(xj)-min(sj), 2*Nx-1);
            Yj = linspace(min(yj)-max(nj), max(yj)-min(nj), 2*Ny-1);

            % Evaluate RS Green's function (forward or inverse)
            if strcmp(method, 'RS')
                H = gRS(Xj, Yj, z_p);
            else
                H = gRS_inverse(Xj, Yj, z_p);
            end

            % Convolve source with Green's function via FFT; scale by pixel areas
            S = ifft2(U_ .* fft2(H)) .* dsj .* dnj;
            % Extract the valid (non-circular) convolution result for this tile
            I(1 + nx*Nx : (nx+1)*Nx, 1 + ny*Ny : (ny+1)*Ny) = S(Nx:end, Ny:end);
        end
    end

    % Crop output to the actual observation plane size (handles non-integer ratio)
    I = I(1:round(index_x * Nx), 1:round(index_y * Ny));

    % --- RS Green's function for forward propagation (+z direction) ---
    % g_RS(x,y,z) = exp(ik*r) * (z/r) / (2*pi*r) * (1/r - ik)
    function out = gRS(x, y, z)
        [X, Y, Z] = meshgrid(x, y, z);
        X = X'; Y = Y'; Z = Z';
        r = sqrt(X.^2 + Y.^2 + Z.^2);
        out = exp(1i.*k.*r).*(z./r)./(2.*pi.*r).*(1./r - 1i.*k);
    end

    % --- RS Green's function for inverse propagation (-z direction, conjugate kernel) ---
    % g_RS_inv(x,y,z) = exp(-ik*r) * (z/r) / (2*pi*r) * (1/r + ik)
    function out = gRS_inverse(x, y, z)
        [X, Y, Z] = meshgrid(x, y, z);
        X = X'; Y = Y'; Z = Z';
        r = sqrt(X.^2 + Y.^2 + Z.^2);
        out = exp(-1i.*k.*r).*(z./r)./(2.*pi.*r).*(1./r + 1i.*k);
    end
end
