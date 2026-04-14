function [s]=gen_mapping(M,D);
% M=2일때 BPSK
% M=4일때 QPSK
% M=16일때 16QAM

rand_I=rand(1,D);
rand_Q=rand(1,D);

if M==2
[s]=2*fix(rand_I*2)-1;

elseif M==4
[s]=2*fix(rand_I*2)-1 + i*(2*fix(rand_Q*2)-1);

else M==16
[s]=-2*fix(rand_I*4)+3 + i*(-2*fix(rand_Q*4)+3);
end;








