
%========by ecl, cbnu ======================%
function [M, m, df] = fftseq(m, ts, df) 
%===========================================%
%		[M,m,df]=fftseq(m,ts,df)
%		[M,m,df]=fftseq(m,ts)
% FFTSEQ		Generates M, the FFT of the sequence m.
%		The sequence is zero padded to meet the required frequency resolution df.
%		ts is the sampling interval. The output df is the final frequency resolution.
%		Output m is the zero padded version of input m. M is the FFT.
%=============================================%
% sampling frequency(Hz)
fs = 1/ts ;

if nargin == 2
  n1 = 0 ;
else
  n1 = fs/df ;
end

% total data number
n2 = length(m) ;
% to find n of n-fft
n = 2^(max(nextpow2(n1), nextpow2(n2))) ;
% Output m : the zero padded version of input m 
m = [m, zeros(1,n-n2)] ;
% return n-fft result of m
M = fft(m,n) ;
% return frequency resolution
df = fs/n ;
%===============================================%