%% plot_two_state_polar_estimation.m
%  Experiment 1 (estimation) only: loads two_state_polar_mse.mat and draws
%     Fig. A  MSE  : 2 x 3  (rows: MSE of eps_T, MSE of phi_T / columns: feedback SNR, phase error, gain ratio)
%     Fig. B  ISR  : 1 x 3  (columns: feedback SNR, phase error, gain ratio)
%     Fig. C  EVM  : 1 x 3  (columns: feedback SNR, phase error, gain ratio)
%  In the phase-error / gain-ratio columns the highest feedback SNR is drawn with solid lines and
%  filled markers, the lower ones with dotted lines and open markers.
%  Requires MATLAB R2016b or later (local functions in a script).
clear; close all; clc;

%% ------------------------------ settings ------------------------------
data_dir = '.';                    % folder that holds two_state_polar_mse.mat
save_fig = true;                   % save every figure
fig_dir  = fullfile(data_dir, 'fig_matlab');
fig_fmt  = {'pdf'};
hide     = {};                     % methods not drawn, e.g. {'huang_flat'}
show_crb = true;                   % CRB in the MSE figure
show_unc = true;                   % uncompensated ISR in the ISR figure
lw = 1.2;  ms = 5;  fs = 9;

% method -> {legend text, colour, marker}
sty = struct();
sty.prop       = {'Proposed',                           [0.85 0 0],    'o'};
sty.huang      = {'Huang et al.',                       [0 0.2 0.9],   's'};
sty.huang_flat = {'Huang et al., subcarrier-averaged',  [0 0.55 0],    '^'};
col_unc = [0 0.65 0.75];

if save_fig && ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ------------------------------ load ------------------------------
f_mse = fullfile(data_dir, 'two_state_polar_mse.mat');
if ~exist(f_mse, 'file'), error('%s not found.', f_mse); end
S = load(f_mse);
methods = strtrim(cellstr(S.methods));  methods = methods(:).';
err_snr = double(S.mse_err_snr(:).');
fprintf('[MSE] methods: %s | trials %d | %d-QAM | sweeps at feedback SNR %s dB\n', ...
    strjoin(methods, ', '), S.trials, S.mse_mod, mat2str(err_snr));

% the three sweeps (= columns of every figure)
cols(1).tags = {'snr'};
cols(1).snr  = NaN;
cols(1).xlab = 'Feedback SNR [dB]';
cols(2).tags = arrayfun(@(v) sprintf('perr%g', v), err_snr, 'UniformOutput', false);
cols(2).snr  = err_snr;
cols(2).xlab = 'State phase error \psi - 90\circ [deg]';
cols(3).tags = arrayfun(@(v) sprintf('gerr%g', v), err_snr, 'UniformOutput', false);
cols(3).snr  = err_snr;
cols(3).xlab = 'State gain ratio \kappa [dB]';

head = sprintf(['%d-QAM data, %d trials; feedback-SNR sweep: state error %g deg / %g dB; ', ...
    'Huang assumes P = %g deg / %g dB'], S.mse_mod, S.trials, S.state_psi_err, S.state_gain_db, ...
    S.huang_p_phase_deg, S.huang_p_gain_db);

%% ------------------------- Fig. A: MSE (2 x 3) -------------------------
% {variable suffix, y label, CRB suffix}
mse_rows = {'mse_eps', 'MSE of \epsilon_T',     'crb_eps'; ...
            'mse_phi', 'MSE of \phi_T [deg^2]', 'crb_phi'};
fig = figure('Name', 'MSE', 'Color', 'w', 'Position', [50 50 1250 650]);
for r = 1:size(mse_rows, 1)
    for c = 1:3
        ax = subplot(2, 3, (r - 1) * 3 + c);
        draw_panel(ax, S, cols(c), mse_rows{r, 1}, methods, hide, sty, err_snr, lw, ms);
        if show_crb, draw_crb(ax, S, cols(c), mse_rows{r, 3}, err_snr, lw); end
        finish_axes(ax, cols(c).xlab, mse_rows{r, 2}, true, fs);
        if r == 1 && c == 1, legend(ax, 'show', 'Location', 'southwest', 'FontSize', fs - 2); end
        if r == 1 && c == 2, legend(ax, 'show', 'Location', 'northwest', 'FontSize', fs - 2); end
    end
end
add_header(fig, ['Estimation MSE, ' head], fs);
save_figure(fig, fig_dir, 'two_state_polar_est_mse', fig_fmt, save_fig);

%% ------------------------- Fig. B: ISR (1 x 3) -------------------------
fig = figure('Name', 'ISR', 'Color', 'w', 'Position', [80 80 1250 380]);
for c = 1:3
    ax = subplot(1, 3, c);
    draw_panel(ax, S, cols(c), 'isr_db', methods, hide, sty, err_snr, lw, ms);
    if show_unc
        tag = cols(c).tags{end};                        % one uncompensated curve per panel
        if isfield(S, [tag '_isr_uncomp_db'])
            plot(ax, double(S.([tag '_x'])(:)), double(S.([tag '_isr_uncomp_db'])(:)), '--', ...
                'Color', col_unc, 'LineWidth', lw, 'DisplayName', 'Uncompensated');
        end
    end
    finish_axes(ax, cols(c).xlab, 'Residual ISR [dB]', false, fs);
    if c == 1 || c == 2, legend(ax, 'show', 'Location', 'best', 'FontSize', fs - 2); end
end
add_header(fig, ['Residual image-to-signal ratio, ' head], fs);
save_figure(fig, fig_dir, 'two_state_polar_est_isr', fig_fmt, save_fig);

%% ------------------------- Fig. C: EVM (1 x 3) -------------------------
fig = figure('Name', 'EVM', 'Color', 'w', 'Position', [110 110 1250 380]);
for c = 1:3
    ax = subplot(1, 3, c);
    draw_panel(ax, S, cols(c), 'res_evm', methods, hide, sty, err_snr, lw, ms);
    finish_axes(ax, cols(c).xlab, 'Residual EVM [%]', true, fs);
    if c == 1 || c == 2, legend(ax, 'show', 'Location', 'best', 'FontSize', fs - 2); end
end
add_header(fig, ['Noise-free residual EVM after compensation, ' head], fs);
save_figure(fig, fig_dir, 'two_state_polar_est_evm', fig_fmt, save_fig);

%% phase change estimated by the proposed method (by-product of the estimator)
if isfield(S, 'snr_mse_psi_prop')
    fig = figure('Name', 'psi estimate', 'Color', 'w', 'Position', [100 100 520 380]);
    semilogy(double(S.snr_x(:)), double(S.snr_mse_psi_prop(:)), '-o', 'Color', sty.prop{2}, ...
        'MarkerFaceColor', sty.prop{2}, 'LineWidth', lw, 'MarkerSize', ms);
    grid on;  set(gca, 'FontSize', fs);
    xlabel('Feedback SNR [dB]');  ylabel('MSE of \psi estimated by the proposed method [deg^2]');
    save_figure(fig, fig_dir, 'two_state_polar_mse_psi', fig_fmt, save_fig);
end

%% --------------------------- local functions ---------------------------
function draw_panel(ax, S, col, var, methods, hide, sty, err_snr, lw, ms)
% one panel: every method, every feedback SNR of this sweep
    hold(ax, 'on');
    for t = 1:numel(col.tags)
        tag = col.tags{t};
        if ~isfield(S, [tag '_x']) || ~isfield(S, [tag '_' var]), continue; end
        x = double(S.([tag '_x'])(:));
        Y = double(S.([tag '_' var]));                      % n_x by n_methods
        if size(Y, 1) ~= numel(x), Y = Y.'; end
        [is_main, suf] = line_kind(col.snr(t), err_snr);
        for j = 1:numel(methods)
            if any(strcmp(methods{j}, hide)) || ~isfield(sty, methods{j}), continue; end
            st = sty.(methods{j});
            if is_main, ls = '-'; mfc = st{2}; else, ls = ':'; mfc = 'none'; end
            plot(ax, x, Y(:, j), 'Color', st{2}, 'LineStyle', ls, 'Marker', st{3}, ...
                'MarkerFaceColor', mfc, 'MarkerIndices', 1:2:numel(x), 'LineWidth', lw, ...
                'MarkerSize', ms, 'DisplayName', [st{1} suf]);
        end
    end
end

function draw_crb(ax, S, col, var, err_snr, lw)
    for t = 1:numel(col.tags)
        tag = col.tags{t};
        if ~isfield(S, [tag '_' var]), continue; end
        [is_main, suf] = line_kind(col.snr(t), err_snr);
        if is_main, ls = '--'; else, ls = '-.'; end
        plot(ax, double(S.([tag '_x'])(:)), double(S.([tag '_' var])(:)), 'k', 'LineStyle', ls, ...
            'LineWidth', lw, 'DisplayName', ['CRB' suf]);
    end
end

function [is_main, suf] = line_kind(snr, err_snr)
% highest feedback SNR (or the feedback-SNR sweep itself) -> solid lines
    is_main = isnan(snr) || snr == max(err_snr);
    if isnan(snr) || numel(err_snr) == 1, suf = '';
    else, suf = sprintf(', %g dB', snr); end
end

function finish_axes(ax, xl, yl, logy, fs)
    if logy, set(ax, 'YScale', 'log'); end
    grid(ax, 'on');  box(ax, 'on');  set(ax, 'FontSize', fs);
    xlabel(ax, xl);  ylabel(ax, yl);
end

function add_header(fig, txt, fs)
    annotation(fig, 'textbox', [0 0.95 1 0.05], 'String', txt, 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'FontSize', fs);
end

function save_figure(fig, fig_dir, name, fmts, enable)
    if ~enable, return; end
    for i = 1:numel(fmts)
        file = fullfile(fig_dir, [name '.' fmts{i}]);
        if exist('exportgraphics', 'file')                       % R2020a or later
            if strcmp(fmts{i}, 'pdf'), exportgraphics(fig, file, 'ContentType', 'vector');
            else, exportgraphics(fig, file, 'Resolution', 300); end
        else
            set(fig, 'PaperPositionMode', 'auto');
            if strcmp(fmts{i}, 'pdf')
                pos = get(fig, 'PaperPosition');  set(fig, 'PaperSize', pos(3:4));
            end
            print(fig, file, ['-d' fmts{i}], '-r300');
        end
    end
end
