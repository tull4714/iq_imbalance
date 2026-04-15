function a=org_cos_sampling(X,Fq)
% X=2일때 2샘플링
% X=4일때 4샘플링...
% Fq=1일 때 주파수 그대로
% Fq=2일 때 주파수 2배
% Fq=3일 때 주파수 3배...

m=1:X;
for t=1:length(m)
    a(t)=cos(2*Fq*pi*t/X);
end
