clear all; clc;

mode1 = 2;  % 1: 송신 original LO, 2: 송신 IQ imbalance 보상
mode2 = 2;   % 1: 수신 original LO, 2: 수신 IQ imbalance 보상

N0 = 64 - 4;    %   Subcarrier number
Block=100;
M=4;    %   Mapping method
D=N0 *Block;   %   Data number
X= 4;    %   Sampling 
Fq=1;   %   Frenquency
Fs = 4; % Sampling Frequency
Fd = Fq;

angle = 5; % deg 
upsilon=0.5;   %   Amplitude
C= angle * pi / 180;    %   Angle
SNR=0:2:15;

a_pilot = [];

%   정보데이터
b_r=[1+1*i -1-1*i];
% pilot = [b_r, fliplr(-conj(b_r))];  % symmetric data-conjugate
%% adjacent data-conjugate
% for m = 1: 2 * length(b_r)
%     if mod(m, 2) == 1
%         pilot(m) = b_r(m);
%     else
%         pilot(m) = -conj(b_r(m - 1));
%     end
% end
pilot = vec2mat(conj([b_r; -conj(b_r)]'), 2 * length(b_r));

l = length(pilot);

[b]=gen_mapping(M,D);

sp= conj(pilot)';
pilot_os = [sp(1: l / 2, :); zeros((Fs / Fd - 1) * l, size(sp, 2)); ...
                sp(l / 2 + 1: l, :)];
pilot_ifft_out=ifft(pilot_os);
ps=vec2mat(pilot_ifft_out',1)';
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
pilot_deos = [rx_fft(1: l / 2, :); ...
                rx_fft(size(rx_fft, 1) - l / 2 + 1: size(rx_fft, 1), :)];
rx_sp= conj(vec2mat(conj(pilot_deos'), 1)');

% pilot per symbol
for m = 1: Block
    a_pilot = [a_pilot sp];
end

%   송신기
N = N0 + l;
sp= conj(vec2mat(b, N0)');
add_z = [a_pilot(1, :); sp(1: N0 / 2, :); a_pilot(2, :); sp(N0 / 2 + 1: N0, :); zeros(l - 2, size(sp, 2))];
os = [add_z(1: N / 2, :); zeros((Fs / Fd - 1) * N, size(add_z, 2)); ...
        add_z(N / 2 + 1: N, :)];
data_ifft_out=ifft(os);

tone = [zeros(N0 + 2, size(data_ifft_out, 2)); a_pilot(3: l, :)];
os_tone = [tone(1: N / 2, :); zeros((Fs / Fd - 1) * N, size(tone, 2)); ...
            tone(N / 2 + 1: N, :)];
pilot_ifft_out = ifft(os_tone);
for m = 0: N - 1
    shift_out = circshift(pilot_ifft_out, 1);
end

ifft_out = data_ifft_out + shift_out;
ps= conj(vec2mat(conj(ifft_out'), 1)');

I_r0 = real(ps);
Q_r0 = imag(ps);

switch mode1
    case 1,
        cos_r=org_cos_sampling(X,Fq);
        sin_r=org_sin_sampling(X,Fq);
        conv_I = conj(I_r0') * cos_r;
        conv_Q = conj(Q_r0') * sin_r;
    case 2,
        [I_r, Q_r] = IQ_comp(ps, X,Fq, b_r(1), rx_sp(1), 'tx');   % IQ 보상
        % IQ LO
        for m = 1: size(I_r, 1)
            m
            conv_I(m, :) = I_r(m, :) .* I_cos;
            conv_Q(m, :) = Q_r(m, :) .* Q_sin;
        end
end

conv_I0 = conj(I_r0') * I_cos;
conv_Q0 = conj(Q_r0') * Q_sin;

conv_Iout_r0 = conj(vec2mat(conv_I0, 1)');
conv_Qout_r0 = conj(vec2mat(conv_Q0, 1)');
conv_Iout_r= conj(vec2mat(conv_I,1)');
conv_Qout_r= conj(vec2mat(conv_Q,1)');

IQ_out_r0 = conv_Iout_r0 + conv_Qout_r0;
IQ_out_r=conv_Iout_r + conv_Qout_r;

y0 = IQ_out_r * Fs / Fd * N;
y= IQ_out_r *Fs/Fd * N;

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
    sigpwr0 = norm(y0) ^ 2 ./ length(y0);
    sigpwr=norm(y)^2./length(y);  %전력공식
    sgma0 = sqrt((Fs / Fd * sigpwr0) / snr_wp);
    sgma=sqrt((Fs / Fd * sigpwr)/snr_wp);     % / 2 / log2(M));
    n0 = sgma0 * randn(1, length(y0));
    n=sgma*randn(1,length(y));
    rx_signal0 = (n0 + y0);
     rx_signal=(n+ y);          
%    rx_signal= y;

%수신기
rx0 = rx_signal0 / (N * Fs / Fd);
rx = rx_signal / (N * Fs/Fd);                    

rx_IQ_out_r0 = vec2mat(rx0, X);
rx_IQ_out_r=vec2mat(rx,X);

    rx_I_out_r0 = (rx_IQ_out_r0 * I_cos')';
    rx_Q_out_r0 = (rx_IQ_out_r0 * Q_sin')';
    rx_Or_I_out_r0 = rx_I_out_r0 * 2 / X;
    rx_Or_Q_out_r0 = rx_Q_out_r0 * 2 / X;
    rx_out_r0 = rx_Or_I_out_r0 + j * rx_Or_Q_out_r0;
    rx_sp_r0 = conj(vec2mat(rx_out_r0, Fs / Fd * N)');
        
    fft_out_r0 = fft(rx_sp_r0);
    
    deos0 = [fft_out_r0(1: N / 2, :); ...
                fft_out_r0(size(fft_out_r0, 1) - N / 2 + 1: size(fft_out_r0, 1), :)];

    de_z0 = [];
    for n = 1: 2
        de_z0 = [de_z0; deos0((N0 / 2 + 1) * (n - 1) + 2: (N0 / 2 + 1) * n, :)];
    end

    act = 1;
    while act
        switch mode2
            case 1,
                down_cos_r=org_cos_sampling(X,Fq);
                down_sin_r=org_sin_sampling(X,Fq);
                rx_I_out_r=(rx_IQ_out_r*down_cos_r')';
                rx_Q_out_r=(rx_IQ_out_r*down_sin_r')';
                act = 0;
            case 2,
                if act == 1
                    rx_I_out_r = (rx_IQ_out_r * I_cos')';
                    rx_Q_out_r = (rx_IQ_out_r * Q_sin')';
                    act = act + 1;
                else
                    for n = 1: size(rx_IQ_out_r, 1)
                        I_r(n, :) = rx_IQ_out_r(n, :) .* I_cos;
                        Q_r(n, :) = rx_IQ_out_r(n, :) .* Q_sin;
                    end
                    [rx_I_out_r, rx_Q_out_r] = IQ_comp(I_r + j * Q_r, X,Fq, b_r(1), de_pilot(1), 'rx');   % IQ 보상
                    act = 0;
                end
            otherwise
                rx_I_out_r = (rx_IQ_out_r * I_cos')';
                rx_Q_out_r = (rx_IQ_out_r * Q_sin')';
                act = 0;
        end

        rx_Or_I_out_r=rx_I_out_r*2/X;
        rx_Or_Q_out_r=rx_Q_out_r*2/X;        
        rx_out_r=rx_Or_I_out_r+rx_Or_Q_out_r*j;
        rx_sp_r= conj(vec2mat(rx_out_r, Fs / Fd * N)');

        fft_out_r=fft(rx_sp_r);

        deos = [fft_out_r(1: N / 2, :); ...
                fft_out_r(size(fft_out_r, 1) - N / 2 + 1: size(fft_out_r, 1), :)];
    
        de_pilot = [];
        de_z = [];
        for n = 1: 2
            de_pilot = [de_pilot; deos((N0 / 2 + 1) * (n - 1) + 1, :)];
            de_z = [de_z; deos((N0 / 2 + 1) * (n - 1) + 2: (N0 / 2 + 1) * n, :)];
        end
    end
%     [I_out, Q_out] = IQ_comp(de_z, 1,Fq, b_r(1), rx_sp(1), 'rx');   % IQ 보상
%     rx_ps_r = conj(vec2mat(conj((I_out + j * Q_out)'), 1)');
    rx_ps_r0 = conj(vec2mat(conj(de_z0'), 1)');
    rx_ps_r= conj(vec2mat(conj(de_z'), 1)');

    
%hard_decision
in_vector=rx_ps_r;
out_hard0 = hard_decision(rx_ps_r0,M);
out_hard=hard_decision(in_vector,M);

%BER
A=out_hard;
B=[b];
ber0(m) = ber_call_qpsk(out_hard0, B, M);
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
semilogy(SNR, ber0, 'g'); hold on;
semilogy(SNR,ber,'r');
axis([0 12 10^-5 1]);