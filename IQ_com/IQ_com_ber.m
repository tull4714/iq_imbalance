clear all; clc;

N= 32;    %   Subcarrier number
Block=100;
M=4;    %   Mapping method
D=N*Block;   %   Data number
X= 4;    %   Sampling 
Fq=1;   %   Frenquency
Fs = 4; % Sampling Frequency
Fd = Fq;

angle = 5; % deg 
upsilon=0.3;   %   Amplitude
C= angle * pi / 180;    %   Angle
SNR=0:2:15;


[b]=gen_mapping(M,D);
%   정보데이터
b_r=[1+1*i -1-1*i];
pilot = [b_r, -(b_r)];
sp= conj(pilot)';
ifft_out=ifft(sp);
% ifft_out = sp;
ps=vec2mat(ifft_out',1)';
I=real(ps);
Q=imag(ps);
[I_cos, Q_sin] = IQ_LO(X,Fq,C,upsilon);
conv_I= conj(I)'* I_cos;
conv_Iout= conj(vec2mat(conv_I,1)');
conv_Q= conj(Q)'* Q_sin;
conv_Qout= conj(vec2mat(conv_Q,1)');
IQ_out=conv_Iout + conv_Qout;
       
%   Feedback 수신기
rx_IQ_out=vec2mat(IQ_out,X);
rx_I_out= conj((rx_IQ_out* conj(I_cos'))');
rx_Q_out= conj((rx_IQ_out* conj(Q_sin'))');
rx_Or_I_out=rx_I_out*2/X;
rx_Or_Q_out=rx_Q_out*2/X;
rx_out=rx_Or_I_out + rx_Or_Q_out *i;
rx_ps= conj(rx_out');
rx_fft=fft(rx_ps);
% rx_fft = rx_ps;
rx_sp=vec2mat(rx_fft',1)';

%   송신기   
sp= conj(vec2mat(b, N)');
os = [sp(1: N / 2, :); zeros((Fs / Fd - 1) * N, size(sp, 2)); sp(N / 2 + 1: N, :)];
ifft_out=ifft(os);
ps= conj(vec2mat(conj(ifft_out'), 1)');

[I_r, Q_r] = IQ_comp(ps, X,Fq, pilot, rx_sp);   % IQ 보상
% I_r = real(ps);
% Q_r = imag(ps);
%% LO
for m = 1: size(I_r, 1)
    conv_I(m, :) = I_r(m, :) .* I_cos;
    conv_Q(m, :) = Q_r(m, :) .* Q_sin;
end

conv_Iout_r= conj(vec2mat(conv_I,1)');
conv_Qout_r= conj(vec2mat(conv_Q,1)');
IQ_out_r=conv_Iout_r + conv_Qout_r;
% y=ps*Fs/Fd * N;                    
% y2 = IQ_out * Fs / Fd * N;

% real_ps = norm(real(ps))^2/length(ps)
% image_ps = norm(imag(ps))^2/length(ps)
% real_cos = norm(up_cos_ra)^2/length(up_cos_ra)
% image_sin =  norm(up_sin_ra)^2/length(up_sin_ra)
% DI=norm(conv_Iout_r)^2/length(conv_Iout_r)
% DQ=norm(conv_Qout_r)^2/length(conv_Qout_r)
% E = sum(conv_Iout_r.*conv_Qout_r)/length(conv_Iout_r)

%AWGN
for m=1:length(SNR)
    snr_wp=10^(SNR(m)/10);  % SNR in decade
    sigpwr=norm(IQ_out_r)^2./length(IQ_out_r);  %전력공식
    sgma=sqrt((Fs / Fd * sigpwr)/snr_wp);     % / 2 / log2(M));
    n=sgma*randn(1,length(IQ_out_r));
     rx_signal=(n+IQ_out_r);          
%    rx_signal=IQ_out_r;
%수신기
% rx = rx_signal / (N * Fs/Fd);                    
rx_IQ_out_r=vec2mat(rx_signal,X);
down_cos_r=org_cos_sampling(X,Fq);
down_sin_r=org_sin_sampling(X,Fq);
rx_I_out_r=(rx_IQ_out_r*down_cos_r')';
rx_Q_out_r=(rx_IQ_out_r*down_sin_r')';
rx_Or_I_out_r=rx_I_out_r*2/X;
rx_Or_Q_out_r=rx_Q_out_r*2/X;
rx_out_r=rx_Or_I_out_r+rx_Or_Q_out_r*j;
rx_sp_r= conj(vec2mat(rx_out_r, Fs / Fd * N)');
fft_out_r=fft(rx_sp_r);
deos = [fft_out_r(1: N / 2, :); ...
        fft_out_r(size(fft_out_r, 1) - N / 2 + 1: size(fft_out_r, 1), :)];
rx_ps_r=vec2mat(deos',1)';

%hard_decision
in_vector=rx_ps_r;
out_hard=hard_decision(in_vector,M);

%BER
A=out_hard;
B=[b];
ber(m) = ber_call_qpsk(A,B,M);
end

% - theory -
for i = 1: length(SNR)
    t_snr = 10 ^ (SNR(i) / 10);
    %theo_err(i) = Q(sqrt(2 * t_snr));
    theo_err(i) = (1 ./ 2) * erfc(sqrt(t_snr)) - (1 ./ 8) * (erfc(sqrt(t_snr))) .^ 2;
end

figure(5)
plot(rx_ps_r,'.'); hold on;
figure(6)
semilogy(SNR, theo_err); hold on;
semilogy(SNR,ber,'r');
axis([0 12 10^-5 1]);