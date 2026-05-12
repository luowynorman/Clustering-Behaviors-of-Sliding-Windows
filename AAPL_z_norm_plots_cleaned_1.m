%% =========================================================================
%  AAPL Stock — Cyclic-Shift Medoid Clustering on Log-Price Windows
%
%  OVERVIEW
%  --------
%  This script implements a multiscale medoid clustering pipeline on
%  z-normalized sliding windows extracted from the log-price series of
%  AAPL stock. The cyclic-shift distance is used as the primary metric,
%  capturing shape similarity up to temporal shift within each window.
%  The pipeline is organized into three cached stages:
%
%    Stage 1 — Distance computation (cached to D_FILE)
%      - Reads closing prices from AAPL.csv and takes log
%      - Extracts all z-normalized sliding windows of length w
%      - Computes the N_wins x N_wins cyclic-shift distance matrix D_cyc
%        via FFT-based cross-correlation
%      - Results saved to D_FILE; reloaded unless RERUN_D = true
%
%    Stage 2 — Multiscale medoid clustering (cached to CLUSTER_FILE)
%      - Initializes k medoids via k-means++ seeding on D_cyc
%      - Iterates: multi-scale histogram top-bin collection ->
%        consecutive-diff ranking -> top-M Voronoi partition ->
%        medoid update -> convergence check (threshold: epsilon = 6)
%      - Results saved to CLUSTER_FILE; reloaded unless RERUN_CLUSTER = true
%      - NOTE: clustering randomness (k-means++ seeding) is controlled
%        by rng(1) at the top of the script. If RERUN_D = false and
%        RERUN_CLUSTER = true, the RNG state at the start of Stage 2
%        depends on the MATLAB session; add rng(SEED_CLUSTER) immediately
%        before the multiscale_medoid call for full reproducibility.
%
%    Stage 3 — MDS embeddings (cached to MDS_FILE)
%      - MDS on the Voronoi-selected submatrix of D_cyc^2
%      - MDS on the full D_cyc^2
%      - MDS on the Euclidean D^2
%      - Results saved to MDS_FILE; reloaded unless RERUN_MDS = true
%
%  VISUALIZATION (Figures 1–6)
%  ---------------------------
%    Figure 1: Voronoi cell members (z-norm waveforms per cluster)
%    Figure 2: k medoid waveforms overlaid
%    Figure 3: k-means centroids on z-normalized windows
%    Figure 4: 2D MDS of Voronoi-selected windows (colored by cluster)
%    Figure 5: 2D MDS of full D_cyc^2 (grey = unselected)
%    Figure 6: 2D MDS of Euclidean D^2 (grey = unselected)
%  All figures saved as .fig and .png to FIG_DIR.
%
%  REPRODUCIBILITY
%  ---------------
%  - Stage 1 (D_cyc): fully deterministic given AAPL.csv, w, TICKER.
%    No randomness involved.
%  - Stage 2 (clustering): k-means++ seeding draws from the global RNG.
%    The top-level rng(1) controls this only when RERUN_D = true (Stage 1
%    consumes no random draws, so the state is preserved). When
%    RERUN_D = false, Stage 1 is skipped and the RNG state is whatever
%    MATLAB initialized at startup. For guaranteed reproducibility across
%    both cases, add rng(SEED_CLUSTER) before multiscale_medoid.
%  - Stage 3 (MDS): fully deterministic given D_cyc and cluster results.
%  - Figure 3 (k-means centroids): not explicitly seeded; not used
%    downstream.
%
%  KEY PARAMETERS
%  --------------
%  w            : sliding window length (default: 100)
%  k            : number of clusters — free to change after Stage 1
%                 (default: 6)
%  M            : Voronoi candidate set size per iteration (default: 600)
%  RERUN_D      : if true, recompute D_cyc even if D_FILE exists
%  RERUN_CLUSTER: if true, rerun clustering even if CLUSTER_FILE exists
%  RERUN_MDS    : if true, rerun MDS even if MDS_FILE exists
%
%  OUTPUT FILES
%  ------------
%  D_FILE       : D_cyc_AAPL_w<w>.mat
%  CLUSTER_FILE : CLUSTER_AAPL_w<w>_k<k>.mat
%  MDS_FILE     : MDS_AAPL_w<w>_k<k>.mat
%  FIG_DIR      : AAPL_figures_k<k>_M<M>/  (contains .fig and .png)
%% =========================================================================
clc; clear; close all;
rng(1)

%% --- User parameters ---
TICKER     = 'AAPL';
w          = 100;       % window length
k          = 10;        % number of clusters (free to change after Stage 1)
M = 600;

RERUN_D       = false;  % true: recompute D_cyc from scratch
RERUN_CLUSTER = true;  % true: rerun medoid clustering
RERUN_MDS     = false;  % true: rerun all MDS embeddings

D_FILE       = sprintf('D_cyc_%s_w%d.mat',    TICKER, w);
CLUSTER_FILE = sprintf('CLUSTER_%s_w%d_k%d.mat', TICKER, w, k);
MDS_FILE     = sprintf('MDS_%s_w%d_k%d.mat',  TICKER, w, k);

%% =========================================================================
%  Stage 1: load price data, build windows, compute D_cyc (once)
%% =========================================================================
if RERUN_D || ~isfile(D_FILE)
    fprintf('=== Stage 1: computing D_cyc ===\n');

    T       = readtable(sprintf('%s.csv', TICKER));
    price   = log(T.Close);
    N_wins  = length(price) - w + 1; % N_wins = number of windows
    znorm   = @(v) (v - mean(v)) / max(std(v), 1e-10);
    windows_raw = zeros(N_wins, w);
    windows_z = zeros(N_wins, w);
    for i = 1:N_wins
        windows_raw(i,:) = price(i : i+w-1);
        windows_z(i,:) = znorm(windows_raw(i,:));
    end

    fprintf('Computing D_cyc (%d x %d) ...\n', N_wins, N_wins);
    F        = fft(windows_z, [], 2);
    norms_sq = sum(windows_z.^2, 2);
    D_cyc    = zeros(N_wins, N_wins);
    for i = 1:N_wins
        xc         = real(ifft(repmat(F(i,:), N_wins, 1) .* conj(F), [], 2));
        vals       = norms_sq(i) + norms_sq - 2*max(xc, [], 2);
        D_cyc(i,:) = sqrt(max(vals, 0))';
        if mod(i, 200) == 0, fprintf('  row %d / %d\n', i, N_wins); end
    end

    save(D_FILE, 'price', 'D_cyc', 'windows_z', 'windows_raw', 'N_wins', 'w', 'TICKER', '-v7.3');
    fprintf('Saved %s\n', D_FILE);
else
    fprintf('=== Stage 1: loading %s ===\n', D_FILE);
    load(D_FILE, 'price', 'D_cyc', 'windows_raw','windows_z', 'N_wins', 'w', 'TICKER');
    fprintf('Loaded.  N_wins=%d, w=%d\n', N_wins, w);
end

%% =========================================================================
%  Stage 2: cluster for chosen k  (cached per k)
%% =========================================================================

if RERUN_CLUSTER || ~isfile(CLUSTER_FILE)
    fprintf('=== Stage 2: clustering  k=%d,  M=%d ===\n', k, M);
    [labels_full, selected, labels_sel, medoid_idx] = ...
        multiscale_medoid(D_cyc, N_wins, k, M);
    save(CLUSTER_FILE, 'labels_full', 'selected', 'labels_sel', 'medoid_idx', 'M', 'k');
    fprintf('Saved %s\n', CLUSTER_FILE);
    % RERUN_MDS = true;   % cluster changed → MDS must be recomputed
else
    fprintf('=== Stage 2: loading %s ===\n', CLUSTER_FILE);
    load(CLUSTER_FILE, 'labels_full', 'selected', 'labels_sel', 'medoid_idx', 'M', 'k');
end

%% =========================================================================
%  Stage 3: MDS embeddings (cached)
%% =========================================================================
norms_sq = sum(windows_z.^2, 2);

if RERUN_MDS || ~isfile(MDS_FILE)
    fprintf('=== Stage 3: computing MDS embeddings ===\n');

    % (a) MDS on selected Voronoi members
    [Y_sel, eigvals_sel, var_exp_sel] = local_mds(D_cyc(selected, selected).^2, 2);
    fprintf('Selected MDS top eigvals: %.4f  %.4f\n', eigvals_sel(1), eigvals_sel(2));

    % (b) MDS on full D_cyc^2
    fprintf('Full D_cyc MDS (%dx%d) ...\n', N_wins, N_wins);
    Y_full = local_mds(D_cyc.^2, 2);

    % (c) MDS on Euclidean D^2
    fprintf('Euclidean MDS ...\n');
    D_euc_sq = max(norms_sq + norms_sq' - 2*(windows_z * windows_z'), 0);
    Y_euc    = local_mds(D_euc_sq, 2);

    save(MDS_FILE, 'Y_sel', 'eigvals_sel', 'var_exp_sel', 'Y_full', 'Y_euc', '-v7.3');
    fprintf('Saved %s\n', MDS_FILE);
else
    fprintf('=== Stage 3: loading %s ===\n', MDS_FILE);
    load(MDS_FILE, 'Y_sel', 'eigvals_sel', 'var_exp_sel', 'Y_full', 'Y_euc');
end

%% =========================================================================
%  Shared colour map
%% =========================================================================
hues = (0:k-1)' / k;
cmap = hsv2rgb([hues, 0.75*ones(k,1), 0.92*ones(k,1)]);

% per-window colour for full-set plots (grey = not selected, cluster colour = selected)
colors_full = 0.82 * ones(N_wins, 3);
for s = 1:k
    colors_full(selected(labels_sel == s), :) = repmat(cmap(s,:), sum(labels_sel==s), 1);
end

%% =========================================================================
%  Figure 1: Voronoi cells (manual centred layout)
%% =========================================================================
n_cols   = ceil(k / 2);
n_bottom = k - n_cols;
margin_x = 0.06; margin_y = 0.10;
pad_x    = 0.04; pad_y    = 0.12;
w_ax     = (1 - 2*margin_x - (n_cols-1)*pad_x) / n_cols;
h_ax     = (1 - 2*margin_y - pad_y) / 2;

figure('Color','w','Position',[80,80,1100,600], ...
       'Name', sprintf('Voronoi Cells — %s  k=%d', TICKER, k));

for s = 1:k
    row = 1 + (s > n_cols);
    col = s - n_cols * (s > n_cols);

    if row == 1
        x0 = margin_x + (col-1) * (w_ax + pad_x);
    else
        total_w = n_bottom * w_ax + (n_bottom-1) * pad_x;
        x0 = (1 - total_w)/2 + (col-1) * (w_ax + pad_x);
    end
    y0 = margin_y + (2 - row) * (h_ax + pad_y);

    ax = axes('Position', [x0, y0, w_ax, h_ax]); %#ok<LAXES>
    hold(ax, 'on');
    cell_idx = selected(labels_sel == s);
    for mi = 1:length(cell_idx)
        plot(ax, 1:w, windows_z(cell_idx(mi),:), 'Color', [cmap(s,:), 0.35], 'LineWidth', 0.8);
    end
    plot(ax, 1:w, windows_z(medoid_idx(s),:), 'k--', 'LineWidth', 2);
    hold(ax,'off');
    title(ax, sprintf('Cluster %d   n=%d', s, length(cell_idx)), 'FontSize', 9);
    xlim(ax,[1 w]); grid(ax,'on'); box(ax,'on');
end

%% =========================================================================
%  Figure 2: k medoids overlaid
%% =========================================================================
figure('Color','w','Position',[100,100,800,200], ...
       'Name', sprintf('k Medoids — %s  k=%d', TICKER, k));
hold on;
for s = 1:k
    plot(1:w, windows_z(medoid_idx(s),:), 'Color', cmap(s,:), 'LineWidth', 2);
end
hold off;
xlabel('offset within window'); ylabel('z-norm');
title(sprintf('%s  —  %d medoids  (cyclic-shift,  w=%d)', TICKER, k, w), ...
      'FontSize', 11, 'FontWeight', 'normal');
xlim([1 w]); grid on; box on;

%% =========================================================================
%  Figure 3: k-means centroids
%% =========================================================================
opts_km   = statset('Display','off','MaxIter',500);
labels_km = kmeans(windows_z, k, 'Replicates', 5, 'Options', opts_km);
centroids = zeros(k, w);
for s = 1:k
    centroids(s,:) = mean(windows_z(labels_km == s, :), 1);
end

figure('Color','w','Position',[100,100,800,200], ...
       'Name', sprintf('k-means Centroids — %s  k=%d', TICKER, k));
hold on;
for s = 1:k
    plot(1:w, centroids(s,:), 'Color', cmap(s,:), 'LineWidth', 2);
end
hold off;
xlabel('offset within window'); ylabel('z-norm');
title(sprintf('%s  —  %d k-means centroids  (z-norm,  w=%d)', TICKER, k, w), ...
      'FontSize', 11, 'FontWeight', 'normal');
xlim([1 w]); grid on; box on;

%% =========================================================================
%  Figure 4: 2D MDS of Voronoi-selected windows
%% =========================================================================
figure('Color','w','Position',[80,80,700,580], ...
       'Name', sprintf('MDS Voronoi — %s  k=%d', TICKER, k));
hold on;
scatter(Y_sel(:,1), Y_sel(:,2), 40, cmap(labels_sel,:), 'filled', 'MarkerFaceAlpha', 0.85);
for s = 1:k
    pos = find(selected == medoid_idx(s), 1);
    if ~isempty(pos)
        % scatter(Y_sel(pos,1), Y_sel(pos,2), 120, cmap(s,:), "p", 'filled', ...
        %         'MarkerEdgeColor','k','LineWidth',1.5);
        scatter(Y_sel(pos,1), Y_sel(pos,2), 180, 'k', 'p', 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        % text(Y_sel(pos,1), Y_sel(pos,2), sprintf('  m_%d',s), ...
        %      'FontSize',9,'Color','k','FontWeight','bold');
    end
end
hold off;
axis equal;
xlabel(sprintf('MDS dim 1  (%.1f%% var)', 100*var_exp_sel(1)), 'FontSize', 11);
ylabel(sprintf('MDS dim 2  (%.1f%% var)', 100*var_exp_sel(2)), 'FontSize', 11);
title(sprintf('%s  —  MDS of Voronoi members  (w=%d, k=%d,  n_{sel}=%d)', ...
      TICKER, w, k, length(selected)), 'FontSize', 11, 'FontWeight', 'normal');
grid on; box on;

%% =========================================================================
%  Figure 5: 2D MDS of full D_cyc^2
%% =========================================================================
figure('Color','w','Position',[80,80,750,620], ...
       'Name', sprintf('Full D^2 MDS — %s  k=%d', TICKER, k));
hold on;
scatter(Y_full(:,1), Y_full(:,2), 8, colors_full, 'filled', 'MarkerFaceAlpha', 0.3);
for s = 1:k
    cell_idx = selected(labels_sel == s);
    scatter(Y_full(cell_idx,1), Y_full(cell_idx,2), 40, cmap(s,:), 'filled', 'MarkerFaceAlpha', 0.8);
end
for s = 1:k
    scatter(Y_full(medoid_idx(s),1), Y_full(medoid_idx(s),2), 150, cmap(s,:), "p", ...
            'filled','MarkerEdgeColor','k','LineWidth',1);
end
hold off;
axis equal;
grid on; box on;

%% =========================================================================
%  Figure 6: 2D MDS of Euclidean D^2
%% =========================================================================
figure('Color','w','Position',[80,80,750,620], ...
       'Name', sprintf('Euclidean D^2 MDS — %s  k=%d', TICKER, k));
hold on;
scatter(Y_euc(:,1), Y_euc(:,2), 8, colors_full, 'filled', 'MarkerFaceAlpha', 0.3);
for s = 1:k
    cell_idx = selected(labels_sel == s);
    scatter(Y_euc(cell_idx,1), Y_euc(cell_idx,2), 40, cmap(s,:), 'filled', 'MarkerFaceAlpha', 0.8);
end
for s = 1:k
    scatter(Y_euc(medoid_idx(s),1), Y_euc(medoid_idx(s),2), 150, cmap(s,:), ...
            'filled','MarkerEdgeColor','k','LineWidth',1.5);
end
hold off;
axis equal;
grid on; box on;

%% =========================================================================
%  Figure 7: k=10 k-means centroids on RAW (un-z-normed) windows
%% =========================================================================
k_raw    = 10;
cmap_raw = hsv2rgb([(0:k_raw-1)'/k_raw, 0.75*ones(k_raw,1), 0.92*ones(k_raw,1)]);

opts_raw      = statset('Display','off','MaxIter',500);
labels_raw    = kmeans(windows_raw, k_raw, 'Replicates', 5, 'Options', opts_raw);
centroids_raw = zeros(k_raw, w);
for s = 1:k_raw
    centroids_raw(s,:) = mean(windows_raw(labels_raw == s, :), 1);
end

figure('Color','w','Position',[100,100,800,200], ...
       'Name', sprintf('k-means Raw Centroids — %s  k=%d', TICKER, k_raw));
hold on;
for s = 1:k_raw
    plot(1:w, centroids_raw(s,:), 'Color', cmap_raw(s,:), 'LineWidth', 2);
end
hold off;
xlabel('time'); ylabel('raw');
title(sprintf('%s  —  %d k-means centroids on raw log-price windows  (w=%d)', ...
      TICKER, k_raw, w), 'FontSize', 11, 'FontWeight', 'normal');
xlim([1 w]);  box on;

%% =========================================================================
%  Figure 8: Original log-price time series
%% =========================================================================
figure('Color','w','Position',[100,100,800,200], ...
       'Name', sprintf('Log Price — %s', TICKER));
plot(price, 'Color', [0.15 0.35 0.65], 'LineWidth', 2.0);
xlabel('trading day index'); ylabel('log(Close)');
title(sprintf('%s  —  log closing price  (N=%d days)', TICKER, length(price)), ...
      'FontSize', 11, 'FontWeight', 'normal');
xlim([1 length(price)]); box on;

%% =========================================================================
%  Save all figures
%% =========================================================================
FIG_DIR = sprintf('AAPL_figures_k%d_M%d', k, M);
if ~isfolder(FIG_DIR), mkdir(FIG_DIR); end

fig_handles = findall(0, 'Type', 'figure');
for f = 1:length(fig_handles)
    fig   = fig_handles(f);
    fname = fullfile(FIG_DIR, matlab.lang.makeValidName(fig.Name));
    savefig(fig, [fname '.fig']);
    exportgraphics(fig, [fname '.png'], 'Resolution', 150);
end
fprintf('All figures saved to "%s/"\n', FIG_DIR);
%% =========================================================================
%  Local function: classical MDS (double-centring + eig)
%  Returns Y (n x dim embedding), top eigenvalues, variance explained
%% =========================================================================
function [Y, eigvals, var_exp] = local_mds(Dsq, dim)
    n           = size(Dsq, 1);
    H           = eye(n) - ones(n)/n;
    B           = -0.5 * H * Dsq * H;
    B           = (B + B') / 2;
    [V, Lambda] = eig(B);
    ev          = diag(Lambda);
    [~, idx]    = sort(abs(ev), 'descend');
    ev          = ev(idx);   V = V(:, idx);
    Y           = V(:,1:dim) .* sqrt(abs(ev(1:dim)))';
    eigvals     = ev(1:dim);
    var_exp     = abs(ev(1:dim)) / sum(abs(ev));
end



 
 
%% =========================================================================
%  Local function: multiscale medoid clustering
%% =========================================================================
function [labels_full, selected, labels_sel, medoid_idx] = ...
         multiscale_medoid(D, N, k, M)
 
    max_iter        = 1000;
    epsilon         = 6;
    n_top_bins_list = [32, 16, 8, 4, 2];
 
    if N <= k
        labels_full = (1:N)';  selected = (1:N)';
        labels_sel  = (1:N)';  medoid_idx = (1:k)';
        return;
    end
 
    % Step 1: k-means++ seeding
    seeds    = zeros(1, k);
    seeds(1) = randi(N);
    for s = 2:k
        d_min    = min(D(:, seeds(1:s-1)), [], 2);
        prob     = d_min.^2 / sum(d_min.^2);
        seeds(s) = randsample(N, 1, true, prob);
    end
 
    selected = []; labels_sel = [];
 
    for iter = 1:max_iter
 
        % Step 2: multi-scale top-bin collection
        top_bin_windows = [];
        for s = 1:k
            prof       = D(seeds(s), :);
            scale_sets = cell(length(n_top_bins_list), 1);
            for sc = 1:length(n_top_bins_list)
                n_top  = n_top_bins_list(sc);
                bw     = 1 / n_top;
                edges  = 0 : bw : (max(prof) + bw);
                [cnt, ~] = histcounts(prof, edges);
                [~, top_bins] = maxk(cnt, min(n_top, numel(cnt)));
                mem = [];
                for b = top_bins
                    mem = [mem, find(prof >= edges(b) & prof < edges(b+1))]; %#ok<AGROW>
                end
                scale_sets{sc} = unique(mem);
            end
            sw = scale_sets{1};
            for sc = 2:length(n_top_bins_list)
                sw = intersect(sw, scale_sets{sc});
            end
            top_bin_windows = union(top_bin_windows, sw);
        end
        top_bin_windows = top_bin_windows(top_bin_windows >= 2);
        if length(top_bin_windows) < k
            top_bin_windows = 1:N;
        end
 
        % Step 3: rank by avg |consecutive diff|, keep top M
        valid = top_bin_windows(top_bin_windows >= 2);
        if length(valid) >= k
            avg_diff = zeros(1, length(valid));
            for s = 1:k
                consec = abs(diff(D(seeds(s),:)));
                avg_diff = avg_diff + consec(valid - 1);
            end
            avg_diff = avg_diff / k;
            [~, sort_idx] = sort(avg_diff, 'ascend');
            selected = valid(sort_idx(1 : min(M, length(valid))));
        else
            selected = (1:N)'; % safeguard when there is too few valid windows (almost never happen)
        end
 
        % Step 4: Voronoi partition of M selected windows
        [~, labels_sel] = min(D(selected, seeds), [], 2);
 
        % Step 5: update medoids
        new_seeds = zeros(1, k);
        for s = 1:k
            members = selected(labels_sel == s);
            if isempty(members)
                new_seeds(s) = seeds(s); continue;
            end
            D_cell       = D(members, members);
            [~, best]    = min(sum(D_cell, 2));
            new_seeds(s) = members(best);
        end
 
        % Step 6: convergence check
        total_shift = sum(arrayfun(@(s) D(new_seeds(s), seeds(s)), 1:k));
        seeds = new_seeds;
        if total_shift < epsilon
            fprintf('multiscale_medoid converged at iteration %d  (total_shift=%.4f)\n', iter, total_shift);
            break;
        end
    end
    if total_shift >= epsilon
        fprintf('multiscale_medoid did NOT converge after %d iterations  (total_shift=%.4f)\n', max_iter, total_shift);
    end
 
    medoid_idx = seeds(:);
    [~, labels_full] = min(D(:, seeds), [], 2);
end