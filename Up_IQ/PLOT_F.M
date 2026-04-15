function [output] = plot_f(input, Fs, df, dB_scale, color, dB_th)
%----------------------------------------------------------------------
%   입력 신호에 대해 FFT를 해서 주파수 영역의 응답을 보여주는 프로그램
%    -> -30dB에서 대역폭을 구할 수 있다.
%                                          
%	Modify 2001. July. 16.	(2000. Nov. 15.)
%   
%	hiber   
%----------------------------------------------------------------------
% plot_f(input, Fs, df,    dB_scale, color, dB_th)
%       input			= Source signal
%       Fs			= frequency of the sample
%       df			= Frequency resolution - 0.1정도
%       dB_scale	= Which of 'Normal' or 'dB' for Scale
%           0         	= normal
%           1(default)	= dB scale
%       color		= color
%		dB_th		= -30dB 대역폭
%----------------------------------------------------------------------

switch nargin
case 3
   dB_scale = 1; color = 'r' ; dB_th = 0;
case 4
   color = 'r' ; dB_th = 0;
case 5
   dB_th = 0;
end

[X_f, x_t, df1] = fftseq(input, 1/Fs, df) ;

X_f1 = X_f / Fs ;

f = [0 : df1 : df1*(length(x_t)-1)]- Fs/2 ;

if dB_scale == 0
   plot( f, fftshift(abs(X_f1)), color ) ;
else
   fft_value = fftshift(20*log10(abs(X_f1)/max(abs(X_f1))));
   plot( f, fft_value, color ) ;
end

cnt=0;
for r = 1 : length(fft_value),
   if fft_value(r) > dB_th
      cnt = cnt + 1;
   else
   end
end
[output] = cnt;
