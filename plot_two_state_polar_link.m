%% plot_two_state_polar_link.m
%  Loads the results of run_iq_imbal_two_state_polar_v4.py and draws the figures.
%     two_state_polar_link.mat  -> one figure per (modulation, channel) + BLER summary per channel
%  Dotted lines without markers = predictions from the residual ISR (v4 results only).
%  Requires MATLAB R2016b or later (local functions in a script).
clear; close all; clc;

%% ------------------------------ settings ------------------------------
data_dir    = '.';                 % folder that holds the two .mat files
save_fig = true;                   % save every figure
fig_dir  = fullfile(data_dir, 'fig_matlab');
fig_fmt  = {'pdf'};

min_blk_err = 1;                   % BLER / coded BER points with fewer block errors are not drawn

show_pred   = true;                % draw the predictions
lw = 1.2;  ms = 5;  fs = 9;

% scheme / method -> {legend text, colour, line style, marker}
sty = struct();
sty.ideal      = {'No I/Q imbalance',                         [0 0 0],          '-',  'none'};
sty.uncomp     = {'Uncompensated',                            [0 0.65 0.75],    '--', 'none'};
sty.prop       = {'Proposed (unknown \psi, \kappa)',          [0.85 0 0],       '-',  'o'};
sty.huang      = {'Huang et al. (assumed P, per subcarrier)', [0 0.2 0.9],      '-',  's'};
sty.huang_flat = {'Huang et al. (assumed P, flat)',           [0 0.55 0],       '--', '^'};

hide = {'huang_flat'};             % 그리지 않을 기법 (비우면 {} )
pred_keys = {'rawber'};            % 예측 점선을 그릴 지표 ('bler', 'ber', 'rawber' 중 선택)
pred_schemes = {'ideal', 'prop'};

if save_fig && ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ========================= experiment 2: link =========================
f_link = fullfile(data_dir, 'two_state_polar_link.mat');
if exist(f_link, 'file')
    L = load(f_link);
    schemes  = strtrim(cellstr(L.schemes));   schemes  = schemes(:).';
    channels = strtrim(cellstr(L.channels));  channels = channels(:).';
    fprintf('[link] polar A=%d, E=%d, N=%d, list %d | feedback SNR %g dB | state error %g deg / %g dB\n', ...
        L.polar_A, L.polar_E, L.polar_N, L.polar_list, L.fb_snr_link, L.state_psi_err, L.state_gain_db);

    % summary figures: selected modulations of one channel drawn together
    sum_mods = [16 64];                                   % 함께 그릴 변조
    info = struct();                                      % 지표 -> {세로축 이름, 로그축 여부, 세로축 범위}
    info.bler   = {'BLER',                            true,  [1e-4 1]};
    info.ber    = {'Coded BER',                       true,  [1e-6 1]};
    info.rawber = {'Uncoded (raw) BER',               true,  [1e-3 1]};
    info.evm    = {'EVM [%]',                         false, []};
    info.thr    = {'Throughput [info bits / symbol]', false, []};
    fig_specs = {{'bler', 'ber', 'rawber'}, 'summary'; ...   % {한 그림에 묶을 지표들, 파일 이름}
        {'evm'},                   'evm'; ...
        {'thr'},                   'thr'};

    for ic = 1:numel(channels)
        for ig = 1:size(fig_specs, 1)
            keys = fig_specs{ig, 1};  nk = numel(keys);
            fig = figure('Name', [fig_specs{ig, 2} ' ' channels{ic}], 'Color', 'w', ...
                'Position', [80 120 430 * nk + 120 480]);
            for ik = 1:nk
                key = keys{ik};  ki = info.(key);
                ax = subplot(1, nk, ik);  hold(ax, 'on');
                first = true;
                for M = sum_mods
                    tag = sprintf('%dQAM_%s', M, channels{ic});
                    if ~isfield(L, ['ebn0_' tag]), continue; end
                    x = double(L.(['ebn0_' tag])(:));
                    for is = 1:numel(schemes)
                        s = schemes{is};  st = sty.(s);
                        if exist('hide', 'var') && any(strcmp(s, hide)), continue; end
                        y = double(L.(sprintf('%s_%s_%s', key, s, tag))(:));
                        if any(strcmp(key, {'bler', 'ber'}))                  % 오류 수가 적은 지점 숨김
                            y(double(L.(sprintf('nblkerr_%s_%s', s, tag))(:)) < min_blk_err) = NaN;
                        end
                        if ki{2}, y(y <= 0) = NaN; end
                        h = plot(ax, x, y, 'Color', st{2}, 'LineStyle', st{3}, 'Marker', st{4}, ...
                            'LineWidth', lw, 'MarkerSize', ms, 'MarkerFaceColor', st{2}, 'DisplayName', st{1});
                        if ~first, set(h, 'HandleVisibility', 'off'); end      % 범례는 첫 변조만
                        pname = sprintf('%s_pred_%s_%s', key, s, tag);          % raw BER: 제안 기법의 이론 곡선
                        if show_pred && any(strcmp(key, pred_keys)) && any(strcmp(s, pred_schemes)) && isfield(L, pname)
                            hp = plot(ax, x, double(L.(pname)(:)), ':', 'Color', st{2}, ...
                                'LineWidth', lw + 0.8, 'DisplayName', 'Theory (proposed)');
                            if ~first, set(hp, 'HandleVisibility', 'off'); end
                        end
                    end
                    yi = double(L.(sprintf('%s_ideal_%s', key, tag))(:));       % ideal 곡선 옆에 변조 이름
                    v = find(yi > 0);
                    if ~isempty(v)
                        k = v(ceil(numel(v) / 2));
                        text(ax, x(k), yi(k), sprintf('  %d-QAM', M), 'FontSize', fs, ...
                            'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
                    end
                    first = false;
                end
                if ki{2}, set(ax, 'YScale', 'log'); end
                if ~isempty(ki{3}), ylim(ax, ki{3}); end
                grid(ax, 'on');  box(ax, 'on');  set(ax, 'FontSize', fs);
                xlabel(ax, 'E_b/N_0 per information bit [dB]');  ylabel(ax, ki{1});
                if ik == 1 || strcmp(key, 'rawber')                             % 범례: 첫 칸과 raw BER 칸
                    legend(ax, 'show', 'Location', 'best', 'FontSize', fs - 2);
                end
            end
            annotation(fig, 'textbox', [0 0.94 1 0.06], 'String', sprintf(['%s, polar (A=%d, E=%d), ', ...
                'feedback SNR %g dB'], channels{ic}, L.polar_A, L.polar_E, L.fb_snr_link), ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', fs);
            save_figure(fig, fig_dir, sprintf('two_state_polar_%s_%s', fig_specs{ig, 2}, channels{ic}), ...
                fig_fmt, save_fig);
        end
    end
else
    warning('%s not found - experiment 2 skipped.', f_link);
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