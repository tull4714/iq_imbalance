function [I_r, Q_r] = IQ_comp(data, X,Fq, pilot0, pilot1, mode)

%   비율계산
% com_p=(real(pilot0) * real(pilot1)- imag(pilot0) * imag(pilot1) ...
%         - (real(pilot0) ^ 2 - imag(pilot0) ^ 2))/ (real(pilot0) ^ 2 + imag(pilot0) ^ 2);
% com_C= asin((2 * real(pilot0) * imag(pilot0) ...
%         - (real(pilot0) * imag(pilot1) + real(pilot1) * imag(pilot0))) ...
%         / (real(pilot0) ^ 2 + imag(pilot0) ^ 2));
com_p=(abs(real(pilot1)) - abs(imag(pilot1))) /2;
com_C= asin((2 - (real(pilot0) .* imag(pilot1) + real(pilot1) .* imag(pilot0))) / 2);

%   진폭보상
a_Ir=2/(2+com_p(1))*real(data);
a_Qr=2/(2-com_p(1))*imag(data);

%   위상보상
m=1:X;
for t=1:length(m)
    S_IC(t) = cos(pi* com_C(1) / 2) ...
                + (tan((2*Fq*pi*t)/X + pi* com_C(1) / 2) * sin(com_C(1) / 2));
    S_QC(t) = cos(pi * com_C(1) / 2) ...
                + (cot((2 * Fq * pi * t) / X - pi * com_C(1) / 2) * sin(com_C(1) / 2));
end

switch mode
    case 'tx'
        I_r = conj(a_Ir)' * S_IC;
        Q_r = conj(a_Qr)' * S_QC;
    case 'rx'
        I_r = a_Ir * S_IC';
        Q_r = a_Qr * S_QC';
end