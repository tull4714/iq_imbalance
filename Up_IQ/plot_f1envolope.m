function [fft_value1]=plot_f1envolope(input, Fs, df, dB_scale, clr, N, fig_num)
%----------------------------------------------------------------------
% plot_f(input, Fs, df,    dB_scale, color, dB_th)
%       input			= Source signal
%       Fs			= frequency of the sample (2*Fs)
%       df			= Frequency resolution : 0.1Á¤µµ
%       dB_scale	= Which of 'Normal' or 'dB' for Scale
%           0         	= normal
%           1(default)	= dB scale
%       color		= color
%		dB_th		= -30dB ´ë¿ªÆø (D/4)
%----------------------------------------------------------------------
[X_f, x_t, df1] = fftseq(input, 1/Fs, df) ;
X_f1 = X_f / Fs ;
 f = [0 : df1 : df1*(length(x_t)-1)]- Fs/2 ;
% f = [0 : df1 : df1*(length(x_t)-1)] ;
if dB_scale == 0
    figure (10),
   plot( f, fftshift(abs(X_f1)), clr ) ;
else
      fft_value = fftshift(20*log10(abs(X_f1)/max(abs(X_f1))));%normalized
      fft_value1 = max(vec2mat(fft_value, N)'); % envelope detect
      f1 = max(vec2mat(f, N)');
      %-----------------------%
      figure(fig_num),
      plot( f1, fft_value1, clr ) ;
      xlabel('f/B');
      ylabel('PSD(dBr)');
%       axis([-4 4 -35 0]); 
end
