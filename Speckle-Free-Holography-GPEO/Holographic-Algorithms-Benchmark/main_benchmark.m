% =========================================================================
% == EXACT LITERATURE-MATCHED HOLOGRAPHIC ALGORITHM COMPARISONS ==
% =========================================================================
% Feature Description:
% 1. GS, DCGS, AWGS use the rigorous Rayleigh-Sommerfeld (RS) propagation 
%    model (RS_FFT_s) to match near-field physical diffraction.
% 2. BCGS [Ref 24] uses a pure fft2 far-field model with energy conservation 
%    to perfectly reproduce the original paper's band-limited methodology.
% 3. Dual-mask design: 
%    - iter_mask: The central Nx*Ny physical region, used as the target 
%      domain during iterative constraints.
%    - eval_mask: The regions where image intensity > 0.9, used as the 
%      objective standard to calculate Speckle Contrast (SC) and Efficiency.
%
% Theoretical / Published Benchmarks for Comparison:
% - MAPS [Ref 26 (Nat. Commun.)] : Efficiency = 29.60%, SC = 0.079
% - Our Work (GPEO proposed)     : Efficiency = 97.00%, SC = 0.013
% =========================================================================
clear; close all; clc;

% =========================================================================
% 1. Initialize Physical and System Parameters
% =========================================================================
disp('--- Initializing Parameters ---');
params.Px = 0.25e-6;         
params.Py = 0.25e-6;         
params.Nx = 512;             
params.Ny = 512;             
params.lambda = 0.532e-6;    
params.zp = 200e-6;          
params.Lx = params.Px * params.Nx;
params.Ly = params.Py * params.Ny;
params.apx = [-params.Lx/2, params.Lx/2];
params.apy = [-params.Ly/2, params.Ly/2];
ratio = 2;
params.obx = ratio * params.apx;
params.oby = ratio * params.apy;
params.Nx_ob = ratio * params.Nx;
params.Ny_ob = ratio * params.Ny;
iterations = 200;

% =========================================================================
% 2. Load Target Image and Generate Padded Masks
% =========================================================================
chemin = 'binary_image.bmp';
disp(['Loading target image from: ', chemin]);
XRGB = imread(chemin);
I_core = imresize(double(XRGB(:,:,1)), [params.Nx, params.Ny]);
I_core = imrotate(I_core, -90);
I_core = sqrt(I_core);             
A_core = I_core ./ max(I_core(:)); 

target_amp = zeros(params.Nx_ob, params.Ny_ob);
start_x = floor((params.Nx_ob - params.Nx) / 2) + 1;
end_x   = start_x + params.Nx - 1;
start_y = floor((params.Ny_ob - params.Ny) / 2) + 1;
end_y   = start_y + params.Ny - 1;
target_amp(start_x:end_x, start_y:end_y) = A_core;

% [Mask 1] iter_mask: Signal domain for the iteration process (central Nx*Ny region)
iter_mask = false(params.Nx_ob, params.Ny_ob);
iter_mask(start_x:end_x, start_y:end_y) = true;

% [Mask 2] eval_mask: The actual signal region (bright target pattern > 0.9) 
% Used exclusively for calculating objective Efficiency and Speckle Contrast (SC).
eval_mask = target_amp > 0.9; 

% =========================================================================
% 3. Run Algorithm Benchmarks
% =========================================================================
disp('--- Starting Algorithm Benchmarks ---');
results = struct('Algorithm', {}, 'Efficiency', {}, 'SC', {});

% --- 1. Traditional GS (RS Propagation) ---
[eff, sc, final_phase_gs, final_image_gs] = run_GS(target_amp, iter_mask, eval_mask, params, iterations);
results(end+1) = struct('Algorithm', 'Traditional GS', 'Efficiency', eff, 'SC', sc);

% --- 2. DCGS [Ref 22] (RS Propagation) ---
[eff, sc, final_phase_dcgs, final_image_dcgs] = run_DCGS(target_amp, iter_mask, eval_mask, params, iterations);
results(end+1) = struct('Algorithm', 'DCGS [Ref 22]', 'Efficiency', eff, 'SC', sc);

% --- 3. BCGS [Ref 24] (Pure FFT Model) ---
[eff, sc, final_phase_bcgs, final_image_bcgs] = run_BCGS(target_amp, iter_mask, eval_mask, params, iterations);
results(end+1) = struct('Algorithm', 'BCGS [Ref 24]', 'Efficiency', eff, 'SC', sc);

% --- 4. AWGS [Ref 25] (RS Propagation) ---
[eff, sc, final_phase_awgs, final_image_awgs] = run_AWGS(target_amp, iter_mask, eval_mask, params, iterations);
results(end+1) = struct('Algorithm', 'AWGS [Ref 25]', 'Efficiency', eff, 'SC', sc);

% =========================================================================
% 4. Print Results and Display Reconstructed Images (Center Region Only)
% =========================================================================
disp('--- Benchmarks Complete ---');
results_table = struct2table(results);
disp(results_table);

% Print reference values from literature for direct comparison
fprintf('\n--- Literature & Proposed Method References ---\n');
fprintf('MAPS [Ref 26]      : Efficiency = 29.60%%, SC = 0.079\n');
fprintf('Our Work (GPEO)    : Efficiency = 97.00%%, SC = 0.013\n');
fprintf('-----------------------------------------------\n');

% Crop the intensity maps to the central physical region (Nx x Ny)
center_image_gs   = final_image_gs(start_x:end_x, start_y:end_y);
center_image_dcgs = final_image_dcgs(start_x:end_x, start_y:end_y);
center_image_bcgs = final_image_bcgs(start_x:end_x, start_y:end_y);
center_image_awgs = final_image_awgs(start_x:end_x, start_y:end_y);

% Plotting and Visualization
figure('Name', 'Final Reconstructed Intensities (Center Region Only)', 'Position',[100, 100, 1200, 300]);

subplot(1, 4, 1); 
imagesc(rot90(center_image_gs, 1)); 
title(sprintf('GS\nEff: %.2f%%, SC: %.3f', results_table.Efficiency(1)*100, results_table.SC(1)));
colorbar;

subplot(1, 4, 2); 
imagesc(rot90(center_image_dcgs, 1)); 
title(sprintf('DCGS\nEff: %.2f%%, SC: %.3f', results_table.Efficiency(2)*100, results_table.SC(2)));
colorbar;

subplot(1, 4, 3); 
imagesc(rot90(center_image_bcgs, 1)); 
title(sprintf('BCGS (FFT)\nEff: %.2f%%, SC: %.3f', results_table.Efficiency(3)*100, results_table.SC(3)));
colorbar;

subplot(1, 4, 4); 
imagesc(rot90(center_image_awgs, 1)); 
title(sprintf('AWGS\nEff: %.2f%%, SC: %.3f', results_table.Efficiency(4)*100, results_table.SC(4)));
colorbar;
colormap(gray);

% =========================================================================
% == Algorithm Internal Implementations ==
% =========================================================================

function [eff, sc, final_phase, final_image_intensity] = run_GS(target_amp, iter_mask, eval_mask, params, iterations)
    fprintf('\nRunning Traditional GS...\n');
    tic;
    holo_field = ones(params.Nx, params.Ny); 
    for iter = 1:iterations
        img_field = RS_FFT_s(holo_field, params.apx, params.apy, params.obx, params.oby, params.zp, params.lambda, 'method', 'RS');
        img_field_constrained = target_amp .* exp(1i * angle(img_field));
        holo_field_new = RS_FFT_s(img_field_constrained, params.obx, params.oby, params.apx, params.apy, params.zp, params.lambda, 'method', 'RS_inverse');
        holo_field = 1.0 .* exp(1i * angle(holo_field_new));
    end
    final_phase = angle(holo_field);
    final_img_field = RS_FFT_s(exp(1i * final_phase), params.apx, params.apy, params.obx, params.oby, params.zp, params.lambda, 'method', 'RS');
    final_image_intensity = abs(final_img_field).^2;
    [eff, sc] = calculate_metrics(final_image_intensity, eval_mask, params);
    fprintf('GS finished in %.2f seconds.\n', toc);
end

function [eff, sc, final_phase, final_image_intensity] = run_DCGS(target_amp, iter_mask, eval_mask, params, iterations)
    fprintf('\nRunning DCGS [Ref 22]...\n');
    tic;
    holo_field = exp(1i * 2 * pi * rand(params.Nx, params.Ny)); 
    beta = 0.5; 
    for iter = 1:iterations
        img_field = RS_FFT_s(holo_field, params.apx, params.apy, params.obx, params.oby, params.zp, params.lambda, 'method', 'RS');
        A_n = abs(img_field); 
        phi_n = angle(img_field); 
        amp_new = A_n;   
        phase_new = phi_n; 
        
        sum_A_S = sum(A_n(iter_mask));
        sum_T_S = sum(target_amp(iter_mask));
        alpha_val = sum_A_S / max(sum_T_S, eps); 
        
        sig_amp = 2 * alpha_val * target_amp(iter_mask) - beta * A_n(iter_mask);
        amp_new(iter_mask) = max(sig_amp, 0); 
        phase_new(iter_mask) = 0; % Force phase to 0 in target region
        
        img_field_constrained = amp_new .* exp(1i * phase_new);
        holo_field_new = RS_FFT_s(img_field_constrained, params.obx, params.oby, params.apx, params.apy, params.zp, params.lambda, 'method', 'RS_inverse');
        holo_field = 1.0 .* exp(1i * angle(holo_field_new));
    end
    final_phase = angle(holo_field);
    final_img_field = RS_FFT_s(exp(1i * final_phase), params.apx, params.apy, params.obx, params.oby, params.zp, params.lambda, 'method', 'RS');
    final_image_intensity = abs(final_img_field).^2;
    [eff, sc] = calculate_metrics(final_image_intensity, eval_mask, params);
    fprintf('DCGS finished in %.2f seconds.\n', toc);
end

function [eff, sc, final_phase, final_image_intensity] = run_BCGS(target_amp, iter_mask, eval_mask, params, iterations)
    fprintf('\nRunning BCGS [Ref 24] with Pure FFT...\n');
    tic;
    [Ny_ob, Nx_ob] = size(target_amp);
    N_total = Ny_ob * Nx_ob; 
    
    holo_mask = zeros(Ny_ob, Nx_ob);
    start_y = floor((Ny_ob - params.Ny) / 2) + 1;
    end_y   = start_y + params.Ny - 1;
    start_x = floor((Nx_ob - params.Nx) / 2) + 1;
    end_x   = start_x + params.Nx - 1;
    holo_mask(start_y:end_y, start_x:end_x) = 1;
    
    k_param = 2 / params.Nx;
    l_param = 2 / params.Ny;
    m_lin = linspace(-Nx_ob/2, Nx_ob/2 - 1, Nx_ob);
    n_lin = linspace(-Ny_ob/2, Ny_ob/2 - 1, Ny_ob);
    [M, N] = meshgrid(m_lin, n_lin);
    
    initial_phase = exp(1i * pi * (k_param * M.^2 + l_param * N.^2));
    if ~isequal(size(initial_phase), size(target_amp)), initial_phase = initial_phase.'; end
    
    img_field_constrained = target_amp .* initial_phase;
    forward_prop  = @(u) fftshift(fft2(ifftshift(u))) / sqrt(N_total);
    backward_prop = @(u) fftshift(ifft2(ifftshift(u))) * sqrt(N_total);
    
    for iter = 1:iterations
        holo_field_full = backward_prop(img_field_constrained);
        holo_field_constrained = holo_mask .* exp(1i * angle(holo_field_full));
        
        img_field = forward_prop(holo_field_constrained);
        amp_new = abs(img_field); 
        amp_new(iter_mask) = target_amp(iter_mask); 
        img_field_constrained = amp_new .* exp(1i * angle(img_field));
    end
    final_img_field = forward_prop(holo_field_constrained);
    final_image_intensity = abs(final_img_field).^2;
    final_phase_full = angle(holo_field_constrained);
    final_phase = final_phase_full(start_y:end_y, start_x:end_x);
    [eff, sc] = calculate_metrics(final_image_intensity, eval_mask, params);
    fprintf('BCGS finished in %.2f seconds.\n', toc);
end

function [eff, sc, final_phase, final_image_intensity] = run_AWGS(target_amp, iter_mask, eval_mask, params, iterations)
    fprintf('\nRunning AWGS [Ref 25]...\n');
    tic;
    holo_field = exp(1i * 2 * pi * ones(params.Nx, params.Ny));
    for iter = 1:iterations
        img_field = RS_FFT_s(holo_field, params.apx, params.apy, params.obx, params.oby, params.zp, params.lambda, 'method', 'RS');
        amp_img = abs(img_field);
        amp_new = amp_img; 
        
        A_t = target_amp(iter_mask);
        A_r = amp_img(iter_mask);
        w_pro = exp(A_t - A_r); 
        amp_new(iter_mask) = A_t .* w_pro; 
        
        img_field_constrained = amp_new .* exp(1i * angle(img_field));
        holo_field_new = RS_FFT_s(img_field_constrained, params.obx, params.oby, params.apx, params.apy, params.zp, params.lambda, 'method', 'RS_inverse');
        holo_field = 1.0 .* exp(1i * angle(holo_field_new));
    end
    final_phase = angle(holo_field);
    final_img_field = RS_FFT_s(exp(1i * final_phase), params.apx, params.apy, params.obx, params.oby, params.zp, params.lambda, 'method', 'RS');
    final_image_intensity = abs(final_img_field).^2;
    [eff, sc] = calculate_metrics(final_image_intensity, eval_mask, params);
    fprintf('AWGS finished in %.2f seconds.\n', toc);
end

function [eff, sc] = calculate_metrics(image_intensity, eval_mask, params)
    signal_power = sum(image_intensity(eval_mask));
    total_power = params.Nx * params.Ny; 
    eff = signal_power / total_power;
    signal_pixels = image_intensity(eval_mask);
    sc = std(signal_pixels) / mean(signal_pixels);
end

% =========================================================================
% == Streamlined RS_FFT_s Core Physics Function ==
% =========================================================================
% Algorithm adapted from:
% "Fast-Fourier-transform based numerical integration method for the
% Rayleigh-Sommerfeld diffraction formula" By Fabin Shen and Anbo Wang
% Streamlined version removing unused methods (Fresnel, FK) and progress bars.
function [I] = RS_FFT_s(src, x_ap, y_ap, x_ob, y_ob, z_p, lambda, varargin)
    method = 'RS';
    for i = 1:2:length(varargin)
        if strcmp(varargin{i}, 'method')
            method = varargin{i+1};
        end
    end
    
    k = 2 * pi / lambda;
    apsizex = max(x_ap) - min(x_ap);
    apsizey = max(y_ap) - min(y_ap);
    obsizex = max(x_ob) - min(x_ob);
    obsizey = max(y_ob) - min(y_ob);
           
    [Nx, Ny] = size(src);
    
    % Divide observation plane into multiple windows for resolution scaling
    index_x = obsizex / apsizex; Nobsx = ceil(index_x);
    index_y = obsizey / apsizey; Nobsy = ceil(index_y);
    
    I = zeros(Nobsx * Nx, Nobsy * Ny);
    
    sj = linspace(min(x_ap), max(x_ap), Nx + 1); sj = sj(1:end-1);
    nj = linspace(min(y_ap), max(y_ap), Ny + 1); nj = nj(1:end-1);
        
    U = zeros(2 * Nx - 1, 2 * Ny - 1);
    U(1:Nx, 1:Ny) = src;
    
    U_ = fft2(U);
    dsj = sj(2) - sj(1);
    dnj = nj(2) - nj(1);
    
    for nx = 0 : Nobsx-1
        for ny = 0 : Nobsy-1
            xj = linspace(min(x_ob) + nx*apsizex, min(x_ob) + (nx+1)*apsizex, Nx+1); xj = xj(1:end-1);
            yj = linspace(min(y_ob) + ny*apsizey, min(y_ob) + (ny+1)*apsizey, Ny+1); yj = yj(1:end-1);
            
            Xj = linspace(min(xj)-max(sj), max(xj)-min(sj), 2*Nx-1);
            Yj = linspace(min(yj)-max(nj), max(yj)-min(nj), 2*Ny-1);
            
            if strcmp(method, 'RS')
                H = gRS(Xj, Yj, z_p);
            else
                H = gRS_inverse(Xj, Yj, z_p);
            end
            
            S = ifft2(U_ .* fft2(H)) .* dsj .* dnj;
            I(1 + nx*Nx : (nx+1)*Nx, 1 + ny*Ny : (ny+1)*Ny) = S(Nx:end, Ny:end);
        end
    end
    
    I = I(1:round(index_x * Nx), 1:round(index_y * Ny));
   
    function out = gRS(x, y, z)
        [X, Y, Z] = meshgrid(x, y, z);
        X = X'; Y = Y'; Z = Z';
        r = sqrt(X.^2 + Y.^2 + Z.^2);
        out = exp(1i.*k.*r).*(z./r)./(2.*pi.*r).*(1./r - 1i.*k);
    end

    function out = gRS_inverse(x, y, z)
        [X, Y, Z] = meshgrid(x, y, z);
        X = X'; Y = Y'; Z = Z';
        r = sqrt(X.^2 + Y.^2 + Z.^2);
        out = exp(-1i.*k.*r).*(z./r)./(2.*pi.*r).*(1./r + 1i.*k);
    end
end
