clc
clear

% 파일 읽기(모든 셀을 문자로)
C = readcell("variable\simul_const.csv", "TextType", "char");  

% 모든 셀에 대해 괄호 제거하고 숫자로 변환 가능한 것만 변환
isChar = cellfun(@ischar, C);
C(isChar) = cellfun(@(s) regexprep(s, '[\(\)]', ''), C(isChar), 'UniformOutput', false);

% 시도해서 숫자로 변환 (변환 불가하면 NaN)
simul_const = cellfun(@(x) str2double(x), C);

% 파일 읽기(모든 셀을 문자로)
C = readcell("variable\blstm_const.csv", "TextType", "char");  

% 모든 셀에 대해 괄호 제거하고 숫자로 변환 가능한 것만 변환
isChar = cellfun(@ischar, C);
C(isChar) = cellfun(@(s) regexprep(s, '[\(\)]', ''), C(isChar), 'UniformOutput', false);

% 시도해서 숫자로 변환 (변환 불가하면 NaN)
blstm_const = cellfun(@(x) str2double(x), C);

figure(5)
plot(simul_const(:, 1: 15000),'.')
grid on

ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.75;      % 그리드 투명도
ax.FontSize = 14;
ax.LabelFontSizeMultiplier = 1.1; % 레이블은 12 * 1.1 = 15.4pt
xlabel('I'); ylabel('Q');
xlim([-2 2]);
ylim([-2 2]);

figure(6)
plot(blstm_const(:, 1: 15000),'.')
grid on

ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.75;      % 그리드 투명도
ax.FontSize = 14;
ax.LabelFontSizeMultiplier = 1.1; % 레이블은 12 * 1.1 = 15.4pt
xlabel('I'); ylabel('Q');
xlim([-2 2]);
ylim([-2 2]);