%% =========================================================================
%  Experiment 2 — CBF Dataset: Cyclic-Shift Medoid Clustering
%
%  OVERVIEW
%  --------
%  This script implements a multiscale medoid clustering pipeline on the
%  CBF (Cylinder-Bell-Funnel) dataset using cyclic-shift distance as the
%  primary metric. The pipeline is organized into three stages:
%
%    Stage 1 — Data generation and distance computation (cached)
%      - Generates the CBF dataset via generate_CBF (seed: SEED_CBF = 42)
%      - Randomly permutes the 90 time series before concatenation
%        (permutation seed is implicitly determined by the RNG state left
%        by generate_CBF after its internal rng(42) call)
%      - Extracts all z-normalized sliding windows of length w from the
%        concatenated series
%      - Computes the N x N cyclic-shift distance matrix D_cyc via FFT
%      - Computes the N x N Euclidean distance matrix D_euc
%      - Computes 2D MDS embeddings of both distance matrices
%      - Results are saved to DATA_FILE and reused unless RERUN_DATA = true
%
%    Stage 2 — Multiscale medoid clustering (cached)
%      - Initializes k medoids via k-means++ seeding on D_cyc
%        (controlled by SEED_CLUSTER for reproducibility)
%      - Iterates: multi-scale histogram top-bin collection ->
%        consecutive-diff ranking -> top-M Voronoi partition ->
%        medoid update -> convergence check (threshold: epsilon)
%      - Results are saved to CLUSTER_FILE and reused unless
%        RERUN_CLUSTER = true
%
%    Stage 3 — Visualization
%      - Figure 1/2: 2D MDS embeddings (cyclic-shift and Euclidean)
%      - Figure 3:   k-means centroids (k=3) vs global mean on Z
%      - Figure 4:   D-profile of a random reference window z_i
%      - Figure 5:   Absolute consecutive difference of that D-profile
%      - Figure 6:   Final D-profile histograms at converged medoids
%      - Figure 7:   Z-normalized waveforms of final Voronoi cell members
%      All figures are saved as .fig and .png to FIG_DIR.
%
%  REPRODUCIBILITY
%  ---------------
%  - Stage 1 (Z, D_cyc, D_euc): fully determined by generate_CBF's
%    internal rng(42), which also fixes the subsequent randperm call.
%    Do not modify generate_CBF or insert any rng() call between
%    generate_CBF and randperm, or the permutation will change.
%  - Stage 2 (clustering): controlled by rng(SEED_CLUSTER) placed
%    immediately before multiscale_medoid. This is independent of
%    whether RERUN_DATA is true or false.
%  - Stage 3 (D-profile reference window z_i): controlled by rng(1)
%    placed immediately before randi(N).
%  - Figure 3 (k-means centroids): not explicitly seeded; results may
%    vary across runs but are not used downstream.
%
%  KEY PARAMETERS
%  --------------
%  w            : sliding window length (default: 128)
%  n_rep        : number of instances per CBF class (default: 30)
%  k            : number of clusters (default: 3)
%  M            : Voronoi candidate set size per iteration (default: 90)
%  epsilon      : medoid convergence threshold in cyclic-shift distance
%                 units (default: 10)
%  SEED_CLUSTER : RNG seed for k-means++ initialization (default: 1)
%  RERUN_DATA   : if true, recompute Stage 1 even if DATA_FILE exists
%  RERUN_CLUSTER: if true, recompute Stage 2 even if CLUSTER_FILE exists
%
%  OUTPUT FILES
%  ------------
%  DATA_FILE    : CBF_data_w<w>_nrep<n_rep>.mat
%  CLUSTER_FILE : CBF_cluster_w<w>_k<k>_M<M>.mat
%  FIG_DIR      : exp2_figures_k<k>_M<M>/  (contains .fig and .png)
%% =========================================================================


%% =========================================================================
clc; clear; close all;
rng(1);

%% --- User parameters ---
w      = 128;
n_rep  = 30;
k      = 3;
M      = 90;        % number of windows kept in Voronoi step
epsilon = 10;        % convergence threshold for medoid shift (default = 10)

class_names = {'Cylinder', 'Bell', 'Funnel'};
cmap        = [0.00 0.40 0.80;
               0.90 0.20 0.10;
               0.10 0.70 0.20];

RERUN_DATA    = false;
RERUN_CLUSTER = true;
SEED_CLUSTER  = 1; % seeding for the iterative k-medoids selection (randomness comes from kmeans++)
DATA_FILE    = sprintf('CBF_data_w%d_nrep%d.mat', w, n_rep);
CLUSTER_FILE = sprintf('CBF_cluster_w%d_k%d_M%d.mat', w, k, M);

%% =========================================================================
%  Stage 1: generate data, compute distances and MDS (cached)
%% =========================================================================
if RERUN_DATA || ~isfile(DATA_FILE)
    fprintf('=== Stage 1: generating data and computing distances ===\n');

    % --- Generate CBF dataset ---
    [~, ~, ~, data_all, labels_all] = generate_CBF(n_rep, w, 42);  % take a random seed on CBF time series generation
    n_total = size(data_all, 1);

    perm       = randperm(n_total);
    data_all   = data_all(perm, :);
    labels_all = labels_all(perm);

    ts = reshape(data_all', [], 1);   % concatenate into one long series

    % --- Z-normalize all sliding windows ---
    znorm    = @(v) (v - mean(v)) / max(std(v), 1e-10);
    N        = length(ts) - w + 1;
    Z        = zeros(N, w);
    for i = 1:N
        Z(i,:) = znorm(ts(i:i+w-1));
    end
    fprintf('Total sliding windows: %d\n', N);

    % Exact boundary positions and their class labels
    exact_idx    = (0:n_total-1)*w + 1;   % 1-based indices into Z
    exact_labels = labels_all;

    % --- Cyclic-shift distance matrix via FFT ---
    fprintf('Computing cyclic-shift D (%dx%d) ...\n', N, N);
    F        = fft(Z, [], 2);
    norms_sq = sum(Z.^2, 2);
    D_cyc    = zeros(N, N, 'single');
    for i = 1:N
        xc         = real(ifft(repmat(F(i,:), N, 1) .* conj(F), [], 2));
        vals       = norms_sq(i) + norms_sq - 2*max(xc, [], 2);
        D_cyc(i,:) = sqrt(single(max(vals, 0)));
        if mod(i, 500) == 0, fprintf('  row %d / %d\n', i, N); end
    end

    % --- Euclidean distance matrix ---
    fprintf('Computing Euclidean D ...\n');
    D_euc = single(sqrt(max(norms_sq + norms_sq' - 2*(Z*Z'), 0)));

    % --- MDS embeddings (2D and 3D) ---
    fprintf('Computing MDS (cyclic-shift) ...\n');
    [Y_cyc2, ev_cyc] = compute_mds(D_cyc, N);

    fprintf('Computing MDS (Euclidean) ...\n');
    [Y_euc2, ev_euc] = compute_mds(D_euc, N);

    save(DATA_FILE, ...
        'w', 'n_rep', 'k', 'class_names', 'cmap', ...
        'data_all', 'labels_all', 'n_total', 'N', ...
        'ts', 'Z', 'D_cyc', 'D_euc', ...
        'exact_idx', 'exact_labels', ...
        'Y_cyc2', 'ev_cyc', ...
        'Y_euc2', 'ev_euc', ...
        '-v7.3');
    fprintf('Saved %s\n', DATA_FILE);
else
    fprintf('=== Stage 1: loading %s ===\n', DATA_FILE);
    load(DATA_FILE);
    fprintf('Loaded.  N=%d, w=%d\n', N, w);
end

%% =========================================================================
%  Stage 2: multiscale medoid clustering (cached per k, M)
%% =========================================================================
if RERUN_CLUSTER || ~isfile(CLUSTER_FILE)
    fprintf('=== Stage 2: clustering  k=%d,  M=%d ===\n', k, M);
    D_full = double(D_cyc);
    rng(SEED_CLUSTER);
    [labels_full, selected, labels_sel, medoid_idx] = ...
        multiscale_medoid(D_full, N, k, M, epsilon);
    save(CLUSTER_FILE, 'labels_full', 'selected', 'labels_sel', 'medoid_idx', 'k', 'M');
    fprintf('Saved %s\n', CLUSTER_FILE);
else
    fprintf('=== Stage 2: loading %s ===\n', CLUSTER_FILE);
    load(CLUSTER_FILE);
    D_full = double(D_cyc);
end

%% =========================================================================
%  Stage 3: Plots
%% =========================================================================

%% --- Figure 1: 2D MDS — cyclic-shift ---
plot_embedding_2d(Y_cyc2, exact_idx, exact_labels, k, cmap, class_names, ...
    'MDS 2D — Cyclic-shift');

%% --- Figure 2: 2D MDS — Euclidean ---
plot_embedding_2d(Y_euc2, exact_idx, exact_labels, k, cmap, class_names, ...
    'MDS 2D — Euclidean');

%% --- Figure 3: k-means centroids (k=3) vs global mean ---
opts_km      = statset('Display', 'off', 'MaxIter', 500);
[~, cents_k3] = kmeans(Z, k, 'Replicates', 10, 'Options', opts_km);
cent_k1      = mean(Z, 1);

colors_k3 = [0.85 0.10 0.10;
             0.10 0.65 0.10;
             0.10 0.40 0.85];

figure('Name', 'K-Means Centroids', 'Color', 'w', 'Position', [100 100 800 200]);
hold on;
for c = 1:k
    plot(1:w, cents_k3(c,:), '-', 'Color', colors_k3(c,:), 'LineWidth', 2.0);
end
plot(1:w, cent_k1, '--', 'Color', [0.55 0.10 0.75], 'LineWidth', 2.0);
yline(0, 'k:', 'LineWidth', 0.8, 'Alpha', 0.4);
hold off;
xlim([1 w]); ylim([-1.3 1.3]);
xlabel('time index'); ylabel('z-normalized amplitude');
title('K-Means Centroids on Z-Normalized Sliding Windows');
box on; grid off;

%% --- Figure 4: D-profile of a random reference window ---
rng(1);
z_i       = randi(N); % randomly choose a window as a reference point
n_side    = 6; % displaying the D-profile curve over [z_i - n_side * w: z_i + n_side * w]
range_lo  = max(1, z_i - n_side * w);
range_hi  = min(N, z_i + n_side * w);
idx_range = range_lo : range_hi;

profile_local     = D_full(z_i, idx_range);
consec_diff_local = abs(diff(profile_local));
idx_range_diff    = idx_range(2:end);

in_range  = exact_idx >= range_lo & exact_idx <= range_hi;
ei_range  = exact_idx(in_range); % collection of exact windows in range
el_range  = exact_labels(in_range);

figure('Name', sprintf('D-profile — reference window z_i=%d', z_i));
hold on;
plot(idx_range, profile_local, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, ...
     'DisplayName', 'd_{cyc}-profile');
for c = 1:k
    ei_c = ei_range(el_range == c);
    if isempty(ei_c), continue; end
    stem(ei_c, D_full(z_i, ei_c), ...
         'Color', cmap(c,:), 'MarkerFaceColor', cmap(c,:), ...
         'MarkerEdgeColor', cmap(c,:), 'MarkerSize', 5, 'LineWidth', 1.0, ...
         'DisplayName', class_names{c});
end
xline(z_i, 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('z_i=%d', z_i));
hold off;
legend('Location', 'northwest', 'FontSize', 9);
xlabel('window index'); ylabel('d_{cyc} distance');
title(sprintf('d_{cyc}-profile  z_i=%d  (range [%d, %d])', z_i, range_lo, range_hi));
xlim([range_lo range_hi]); grid on; box on;

%% --- Figure 5: Absolute consecutive difference of D-profile ---
figure('Name', sprintf('Consecutive diff — z_i=%d', z_i));
hold on;
plot(idx_range_diff, consec_diff_local, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, ...
     'DisplayName', '|diff d_{cyc}|');
for c = 1:k
    ei_c  = ei_range(el_range == c);
    ei_c  = ei_c(ei_c >= range_lo + 1);
    if isempty(ei_c), continue; end
    d_vals = abs(D_full(z_i, ei_c) - D_full(z_i, ei_c - 1));
    stem(ei_c, d_vals, ...
         'Color', cmap(c,:), 'MarkerFaceColor', cmap(c,:), ...
         'MarkerEdgeColor', cmap(c,:), 'MarkerSize', 5, 'LineWidth', 1.0, ...
         'DisplayName', class_names{c});
end
xline(z_i, 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('z_i=%d', z_i));
hold off;
legend('Location', 'northwest', 'FontSize', 9);
xlabel('window index'); ylabel('|d_{cyc}(z_j,z_i) - d_{cyc}(z_{j-1},z_i)|');
title(sprintf('Consecutive diff of d_{cyc}-profile  z_i=%d', z_i));
xlim([range_lo range_hi]); grid on; box on;

%% --- Figure 6: Final D-profile histograms with top bins highlighted ---
n_top_bins_display = 2;
bin_width_display  = 1 / n_top_bins_display;

figure('Name', 'Final D-profile Histograms');
for s = 1:k
    subplot(k, 1, s);
    hold on;

    prof        = D_full(medoid_idx(s), :);
    edges       = 0 : bin_width_display : (max(prof) + bin_width_display);
    counts      = histcounts(prof, edges);
    bin_centers = edges(1:end-1) + bin_width_display/2;

    [~, top_bins] = maxk(counts, n_top_bins_display);
    bar_colors    = repmat([0.75 0.75 0.75], numel(bin_centers), 1);
    bar_colors(top_bins, :) = repmat([1.0 0.85 0.0], n_top_bins_display, 1);

    b        = bar(bin_centers, counts, 1, 'FaceColor', 'flat', 'EdgeColor', 'none');
    b.CData  = bar_colors;

    for c = 1:k
        ei    = exact_idx(exact_labels == c);
        d_v   = prof(ei);
        bin_id = min(floor(d_v / bin_width_display) + 1, numel(counts));
        stem(d_v, counts(bin_id), ...
             'Color', cmap(c,:), 'MarkerFaceColor', cmap(c,:), ...
             'MarkerEdgeColor', cmap(c,:), 'MarkerSize', 5, 'LineWidth', 0.8, ...
             'DisplayName', class_names{c});
    end

    hold off;
    legend([{sprintf('Top %d bins', n_top_bins_display)}, class_names], ...
           'Location', 'best', 'FontSize', 9);
    xlabel('cyclic-shift distance'); ylabel('window count');
    title(sprintf('Medoid %d  (window %d,  bin width=%.3f)', ...
                  s, medoid_idx(s), bin_width_display));
    grid on; box on;
end

%% --- Figure 7: Final Voronoi cell members ---
figure('Name', 'Final Voronoi Cell Members', 'Color', 'w');
for s = 1:k
    subplot(k, 1, s);
    hold on;
    cell_idx = selected(labels_sel == s); % the indices of all Voronoi cell members of medoid s
    for mi = 1:length(cell_idx)
        plot(Z(cell_idx(mi),:), 'Color', [cmap(s,:), 0.25], 'LineWidth', 0.5); % plot cell members
    end
    plot(Z(medoid_idx(s),:), 'k--', 'LineWidth', 2.0); % plot medoid s
    hold off;
    title(sprintf('Cluster %d  (medoid window %d)  —  n=%d', ...
                  s, medoid_idx(s), length(cell_idx)));
    xlabel('time'); ylabel('z-norm');
    xlim([1 w]); grid on; box on;
end

%% =========================================================================
%  Save all figures
%% =========================================================================
FIG_DIR = sprintf('exp2_figures_k%d_M%d', k, M);
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
%  Helper: 2D MDS scatter with exact windows highlighted
%% =========================================================================
function plot_embedding_2d(Y, exact_idx, exact_labels, k, cmap, class_names, ttl)
    figure('Name', ttl, 'Color', 'w');
    hold on;
    scatter(Y(:,1), Y(:,2), 8, [0.75 0.75 0.75], 'filled', ...
            'MarkerFaceAlpha', 0.25, 'DisplayName', 'All windows');
    for c = 1:k
        ei = exact_idx(exact_labels == c);
        scatter(Y(ei,1), Y(ei,2), 120, cmap(c,:), 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 0.8, ...
                'DisplayName', class_names{c});
    end
    hold off;
    legend('Location', 'northwest', 'FontSize', 10);
    xlabel('dim 1'); ylabel('dim 2');
    title(ttl);
    axis equal;
    ax = gca;
    lo = min([ax.XLim, ax.YLim]);
    hi = max([ax.XLim, ax.YLim]);
    xlim([lo hi]); ylim([lo hi]);
    grid on; box on;
end

%% =========================================================================
%  Helper: MDS via sparse eigs (fast for large N)
%% =========================================================================
function [Y2, lambda_vals] = compute_mds(Dist, N)
    H           = eye(N) - ones(N,N)/N;
    B           = -0.5 * H * double(Dist).^2 * H;
    B           = (B + B') / 2;
    [V, Lambda] = eigs(B, 2, 'largestreal');
    lambda_vals = real(diag(Lambda));
    lp          = max(lambda_vals, 0);
    Y2          = V(:,1:2) * diag(sqrt(lp(1:2)));
end

%% =========================================================================
%  Local function: multiscale medoid clustering
%% =========================================================================
function [labels_full, selected, labels_sel, medoid_idx] = ...
         multiscale_medoid(D, N, k, M, epsilon)

    max_iter        = 1000;
    n_top_bins_list = [32, 16, 8, 4, 2];

    if N <= k
        labels_full = (1:N)';  selected = (1:N)';
        labels_sel  = (1:N)';  medoid_idx = (1:k)';
        return;
    end

    % Step 1: initialization with k-means++ seeding 
    seeds    = zeros(1, k);
    seeds(1) = randi(N);
    for s = 2:k
        d_min    = min(D(:, seeds(1:s-1)), [], 2);
        prob     = d_min.^2 / sum(d_min.^2);
        seeds(s) = randsample(N, 1, true, prob);
    end

    selected = []; labels_sel = []; total_shift = Inf;

    for iter = 1:max_iter

        % Step 2: multi-scale top-bin collection
        top_bin_windows = [];
        for s = 1:k
            prof       = D(seeds(s), :);
            scale_sets = cell(length(n_top_bins_list), 1);
            for sc = 1:length(n_top_bins_list)
                n_top = n_top_bins_list(sc);
                bw    = 1 / n_top;
                edges = 0 : bw : (max(prof) + bw);
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
                consec   = abs(diff(D(seeds(s),:)));
                avg_diff = avg_diff + consec(valid - 1);
            end
            avg_diff = avg_diff / k;
            [~, sort_idx] = sort(avg_diff, 'ascend');
            selected = valid(sort_idx(1 : min(M, length(valid))));
        else
            selected = (1:N)';
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

%% =========================================================================
%  Local function: generate_CBF
%% =========================================================================
function [cylinder, bell, funnel, dataset, labels] = generate_CBF(N_per_class, w, seed)
    if nargin < 1, N_per_class = 30;  end
    if nargin < 2, w           = 128; end
    if nargin < 3, seed        = 42;  end
    rng(seed);

    cylinder = zeros(N_per_class, w);
    bell     = zeros(N_per_class, w);
    funnel   = zeros(N_per_class, w);

    for ii = 1:N_per_class
        a = randi([16,32]); b = min(a + randi([32,96]), w);
        chi = zeros(1,w); chi(a:b) = 1;
        cylinder(ii,:) = (6 + randn)*chi + randn(1,w);
    end
    for ii = 1:N_per_class
        a = randi([16,32]); b = min(a + randi([32,96]), w);
        chi = zeros(1,w); ramp = zeros(1,w);
        for jj = a:b, chi(jj) = 1; ramp(jj) = (jj-a)/(b-a); end
        bell(ii,:) = (6 + randn)*chi.*ramp + randn(1,w);
    end
    for ii = 1:N_per_class
        a = randi([16,32]); b = min(a + randi([32,96]), w);
        chi = zeros(1,w); ramp = zeros(1,w);
        for jj = a:b, chi(jj) = 1; ramp(jj) = (b-jj)/(b-a); end
        funnel(ii,:) = (6 + randn)*chi.*ramp + randn(1,w);
    end

    dataset = [cylinder; bell; funnel];
    labels  = [ones(N_per_class,1); 2*ones(N_per_class,1); 3*ones(N_per_class,1)];
end
