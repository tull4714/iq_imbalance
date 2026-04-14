function [output] = add_pilot(data, pilot, N, N0)

% for ¹® add_z case 1:
cnt = 1;
for m = 1: N - 2
    if m == (N0 / 2) * (cnt - 1) + cnt
        output1(m, :) = pilot(cnt, :);
        cnt = cnt + 1;
    else
        output1(m, :) = data(m - (cnt - 1), :)
    end
end

% for ¹® add_z case 2:
output2 = [];
for m = 1: 8
    if mod(m, 2) == 1
        output2 = [output2; pilot((m + 1) / 2, :)];
    else
        output2 = [output2; data((N0 / 4) * (m / 2 - 1) + 1: (N0 / 4) * (m / 2), :)];
    end
end