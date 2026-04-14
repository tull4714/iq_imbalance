function a=cos_sampling_a(X,Fq,C,upsilon,com_C)
% X=2일때 2샘플링
% X=4일때 4샘플링...
% Fq=1일 때 주파수 그대로
% Fq=2일 때 주파수 2배
% Fq=3일 때 주파수 3배...

m=1:X;
for t=1:length(m)
    a(t)=(1+upsilon/2)*cos((2*Fq*pi*t)/(X+pi*C/360-pi*com_C/360));
end

