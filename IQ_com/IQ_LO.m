function [I_LO, Q_LO] = IQ_LO(X,Fq,C, epsilon)
% X=2일때 2샘플링
% X=4일때 4샘플링...
% Fq=1일 때 주파수 그대로
% Fq=2일 때 주파수 2배
% Fq=3일 때 주파수 3배...

m=1:X;
for t=1:length(m)
    I_LO(t)=(1+ epsilon/2)*cos((2*Fq*pi*t)/X+ C/ 2);
    Q_LO(t) = (1 - epsilon / 2) * sin((2 * pi * Fq * t) / X - C / 2);
end