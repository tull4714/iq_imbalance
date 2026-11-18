%% plot_two_state_polar.m
%  Loads the results of run_iq_imbal_two_state_polar_v4.py and draws the figures.
%     two_state_polar_mse.mat   -> Fig. 1 (estimation MSE / residual ISR / residual EVM)
%     two_state_polar_link.mat  -> one figure per (modulation, channel) + BLER summary per channel
%  Dotted lines without markers = predictions from the residual ISR (v4 results only).
%  Requires MATLAB R2016b or later (local functions in a script).
clear; close all; clc;

%% ------------------------------ settings ------------------------------
data_dir    = '.';                 % folder that holds the two .mat files
save_fig    = true;                % save every figure
fig_dir     = fullfile(data_dir, 'fig_matlab');
fig_fmt     = {'png', 'pdf'};
min_blk_err = 1;                   % BLER / coded BER points with fewer block errors are not drawn
                                   % (use e.g. 10 to hide statistically weak points)
show_pred   = true;                % draw the predictions
lw = 1.2;  ms = 5;  fs = 9;

% scheme / method -> {legend text, colour, line style, marker}
sty = struct();
sty.ideal      = {'No I/Q imbalance',                         [0 0 0],          '-',  'none'};
sty.uncomp     = {'Uncompensated',                            [0 0.65 0.75],    '--', 'none'};
sty.prop       = {'Proposed (unknown \psi, \kappa)',          [0.85 0 0],       '-',  'o'};
sty.huang      = {'Huang et al. (assumed P, per subcarrier)', [0 0.2 0.9],      '-',  's'};
sty.huang_flat = {'Huang et al. (assumed P, flat)',           [0 0.55 0],       '--', '^'};

if save_fig && ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ====================== experiment 1: estimation ======================
f_mse = fullfile(data_dir, 'two_state_polar_mse.mat');
if exist(f_mse, 'file')
    S = load(f_mse);
    methods = strtrim(cellstr(S.methods));  methods = methods(:).';
    err_snr = double(S.mse_err_snr(:).');
    fprintf('[MSE] methods: %s | trials %d | %d-QAM | sweeps at feedback SNR %s dB\n', ...
        strjoin(methods, ', '), S.trials, S.mse_mod, mat2str(err_snr));

    % columns: feedback SNR sweep, state phase error sweep, state gain ratio sweep
    col_tags = {{'snr'}, arrayfun(@(v) sprintf('perr%g', v), err_snr, 'UniformOutput', false), ...
                         arrayfun(@(v) sprintf('gerr%g', v), err_snr, 'UniformOutput', false)};
    col_snr  = {NaN, err_snr, err_snr};
    xlab = {'Feedback SNR [dB]', 'State phase error \psi - 90\circ [deg]', 'State gain ratio \kappa [dB]'};
    rows = {'mse_eps', 'MSE of \epsilon_T',           true,  'crb_eps'; ...
            'mse_phi', 'MSE of \phi_T [deg^2]',       true,  'crb_phi'; ...
            'isr_db',  'Residual ISR [dB]',           false, ''; ...
            'res_evm', 'Residual EVM [%]',            true,  ''};

    fig = figure('Name', 'Estimation', 'Color', 'w', 'Position', [50 50 1250 1000]);
    for c = 1:3
        for r = 1:size(rows, 1)
            ax = subplot(size(rows, 1), 3, (r - 1) * 3 + c);  hold(ax, 'on');
            for t = 1:numel(col_tags{c})
                tag = col_tags{c}{t};
                if ~isfield(S, [tag '_x']), continue; end
                x = double(S.([tag '_x'])(:));
                Y = double(S.([tag '_' rows{r, 1}]));               % n_x by n_methods
                if size(Y, 1) ~= numel(x), Y = Y.'; end
                is_main = isnan(col_snr{c}(t)) || col_snr{c}(t) == max(err_snr);
                if isnan(col_snr{c}(t)) || numel(err_snr) == 1, suf = '';
                else, suf = sprintf(', %g dB', col_snr{c}(t)); end
                for j = 1:numel(methods)
                    st = sty.(methods{j});
                    h = plot(ax, x, Y(:, j), 'Color', st{2}, 'Marker', st{4}, 'LineWidth', lw, ...
                        'MarkerSize', ms, 'DisplayName', [st{1} suf]);
                    if is_main, set(h, 'LineStyle', '-', 'MarkerFaceColor', st{2});
                    else,       set(h, 'LineStyle', ':', 'MarkerFaceColor', 'none'); end
                    set(h, 'MarkerIndices', 1:2:numel(x));
                end
                if ~isempty(rows{r, 4}) && isfield(S, [tag '_' rows{r, 4}])
                    crb = double(S.([tag '_' rows{r, 4}])(:));
                    if is_main, ls = '--'; else, ls = '-.'; end
                    plot(ax, x, crb, 'k', 'LineStyle', ls, 'LineWidth', lw, 'DisplayName', ['CRB' suf]);
                end
                if strcmp(rows{r, 1}, 'isr_db') && is_main && isfield(S, [tag '_isr_uncomp_db'])
                    st = sty.uncomp;
                    plot(ax, x, double(S.([tag '_isr_uncomp_db'])(:)), 'Color', st{2}, 'LineStyle', st{3}, ...
                        'LineWidth', lw, 'DisplayName', st{1});
                end
            end
            if rows{r, 3}, set(ax, 'YScale', 'log'); end
            grid(ax, 'on');  box(ax, 'on');  set(ax, 'FontSize', fs);
            xlabel(ax, xlab{c});  ylabel(ax, rows{r, 2});
            if r == 3, legend(ax, 'show', 'Location', 'best', 'FontSize', fs - 2); end
        end
    end
    annotation(fig, 'textbox', [0 0.965 1 0.035], 'String', sprintf(['Two-state estimation, %d-QAM data, ', ...
        '%d trials; column 1: state error %g deg / %g dB; Huang assumes P = %g deg / %g dB'], S.mse_mod, ...
        S.trials, S.state_psi_err, S.state_gain_db, S.huang_p_phase_deg, S.huang_p_gain_db), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', fs);
    save_figure(fig, fig_dir, 'two_state_polar_mse', fig_fmt, save_fig);

    % phase change estimated by the proposed method (by-product of the estimator)
    if isfield(S, 'snr_mse_psi_prop')
        fig = figure('Name', 'psi estimate', 'Color', 'w', 'Position', [100 100 520 380]);
        semilogy(double(S.snr_x(:)), double(S.snr_mse_psi_prop(:)), '-o', 'Color', sty.prop{2}, ...
            'MarkerFaceColor', sty.prop{2}, 'LineWidth', lw, 'MarkerSize', ms);
        grid on;  set(gca, 'FontSize', fs);
        xlabel('Feedback SNR [dB]');  ylabel('MSE of \psi estimated by the proposed method [deg^2]');
        save_figure(fig, fig_dir, 'two_state_polar_mse_psi', fig_fmt, save_fig);
    end
else
    warning('%s not found - experiment 1 skipped.', f_mse);
end

%% ========================= experiment 2: link =========================
f_link = fullfile(data_dir, 'two_state_polar_link.mat');
if exist(f_link, 'file')
    L = load(f_link);
    schemes  = strtrim(cellstr(L.schemes));   schemes  = schemes(:).';
    channels = strtrim(cellstr(L.channels));  channels = channels(:).';
    mods = double(L.mod_list(:).');
    fprintf('[link] polar A=%d, E=%d, N=%d, list %d | feedback SNR %g dB | state error %g deg / %g dB\n', ...
        L.polar_A, L.polar_E, L.polar_N, L.polar_list, L.fb_snr_link, L.state_psi_err, L.state_gain_db);

    panels = {'bler',   'BLER',                              true,  [1e-4 1]; ...
              'ber',    'Coded BER',                         true,  [1e-6 1]; ...
              'rawber', 'Uncoded (raw) BER',                 true,  []; ...
              'evm',    'EVM [%]',                           false, []; ...
              'thr',    'Throughput [info bits / symbol]',   false, []; ...
              'snrloss_pred', 'Predicted SNR loss vs ideal [dB]', false, []};

    for M = mods
        for ic = 1:numel(channels)
            tag = sprintf('%dQAM_%s', M, channels{ic});
            if ~isfield(L, ['ebn0_' tag]), continue; end
            x = double(L.(['ebn0_' tag])(:));
            fig = figure('Name', tag, 'Color', 'w', 'Position', [60 60 1300 720]);
            for p = 1:size(panels, 1)
                key = panels{p, 1};
                ax = subplot(2, 3, p);  hold(ax, 'on');
                for is = 1:numel(schemes)
                    s = schemes{is};  st = sty.(s);
                    name = sprintf('%s_%s_%s', key, s, tag);
                    if ~isfield(L, name), continue; end
                    if strcmp(key, 'snrloss_pred') && strcmp(s, 'ideal'), continue; end
                    y = double(L.(name)(:));
                    if any(strcmp(key, {'bler', 'ber'}))                 % hide unreliable points
                        y(double(L.(sprintf('nblkerr_%s_%s', s, tag))(:)) < min_blk_err) = NaN;
                    end
                    if panels{p, 3}, y(y <= 0) = NaN; end
                    plot(ax, x, y, 'Color', st{2}, 'LineStyle', st{3}, 'Marker', st{4}, 'LineWidth', lw, ...
                        'MarkerSize', ms, 'MarkerFaceColor', st{2}, 'DisplayName', st{1});
                    pname = sprintf('%s_pred_%s_%s', key, s, tag);       % prediction: dotted, no marker
                    if show_pred && isfield(L, pname) && ~(strcmp(s, 'ideal') && ~strcmp(key, 'rawber'))
                        yp = double(L.(pname)(:));  yp(yp <= 0) = NaN;
                        plot(ax, x, yp, ':', 'Color', st{2}, 'LineWidth', lw + 0.8, 'HandleVisibility', 'off');
                    end
                end
                if panels{p, 3}, set(ax, 'YScale', 'log'); end
                if ~isempty(panels{p, 4}), ylim(ax, panels{p, 4}); end
                grid(ax, 'on');  box(ax, 'on');  set(ax, 'FontSize', fs);
                xlabel(ax, 'E_b/N_0 per information bit [dB]');  ylabel(ax, panels{p, 2});
                if p == 1
                    if show_pred && isfield(L, sprintf('bler_pred_prop_%s', tag))
                        plot(ax, NaN, NaN, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', lw + 0.8, ...
                            'DisplayName', 'Prediction from ISR');
                    end
                    legend(ax, 'show', 'Location', 'southwest', 'FontSize', fs - 2);
                end
            end
            annotation(fig, 'textbox', [0 0.955 1 0.045], 'String', sprintf(['%d-QAM, %s, polar (A=%d, E=%d, ', ...
                'CA-SCL L=%d), feedback SNR %g dB, state error %g deg / %g dB, Huang P = %g deg / %g dB, %d slots'], ...
                M, channels{ic}, L.polar_A, L.polar_E, L.polar_list, L.fb_snr_link, L.state_psi_err, ...
                L.state_gain_db, L.huang_p_phase_deg, L.huang_p_gain_db, L.(['frames_' tag])), ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', fs);
            save_figure(fig, fig_dir, ['two_state_polar_link_' tag], fig_fmt, save_fig);

            % residual ISR in the link, printed for reference
            for is = 1:numel(schemes)
                name = sprintf('isr_db_%s_%s', schemes{is}, tag);
                if isfield(L, name), fprintf('   %-16s ISR of %-10s: %7.1f dB\n', tag, schemes{is}, L.(name)); end
            end
        end
    end

    % BLER summary: all modulations of one channel in one axes
    for ic = 1:numel(channels)
        fig = figure('Name', ['BLER ' channels{ic}], 'Color', 'w', 'Position', [120 120 760 520]);
        ax = axes(fig);  hold(ax, 'on');
        for im = 1:numel(mods)
            tag = sprintf('%dQAM_%s', mods(im), channels{ic});
            if ~isfield(L, ['ebn0_' tag]), continue; end
            x = double(L.(['ebn0_' tag])(:));
            for is = 1:numel(schemes)
                s = schemes{is};  st = sty.(s);
                y = double(L.(sprintf('bler_%s_%s', s, tag))(:));
                y(double(L.(sprintf('nblkerr_%s_%s', s, tag))(:)) < min_blk_err | y <= 0) = NaN;
                h = plot(ax, x, y, 'Color', st{2}, 'LineStyle', st{3}, 'Marker', st{4}, 'LineWidth', lw, ...
                    'MarkerSize', ms, 'MarkerFaceColor', st{2}, 'DisplayName', st{1});
                if im > 1, set(h, 'HandleVisibility', 'off'); end
            end
            k = find(double(L.(sprintf('bler_ideal_%s', tag))(:)) < 0.5, 1);     % label next to the curve
            if ~isempty(k), text(ax, x(k), 0.6, sprintf('%d-QAM', mods(im)), 'FontSize', fs, ...
                    'HorizontalAlignment', 'center', 'BackgroundColor', 'w'); end
        end
        set(ax, 'YScale', 'log', 'FontSize', fs);  ylim(ax, [1e-4 1]);  grid(ax, 'on');  box(ax, 'on');
        xlabel(ax, 'E_b/N_0 per information bit [dB]');  ylabel(ax, 'BLER');
        title(ax, sprintf('%s, polar (A=%d, E=%d), feedback SNR %g dB', channels{ic}, L.polar_A, L.polar_E, ...
            L.fb_snr_link), 'FontWeight', 'normal');
        legend(ax, 'show', 'Location', 'southwest', 'FontSize', fs - 1);
        save_figure(fig, fig_dir, ['two_state_polar_bler_' channels{ic}], fig_fmt, save_fig);
    end
else
    warning('%s not found - experiment 2 skipped.', f_link);
end

%% --------------------------- local functions ---------------------------
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
