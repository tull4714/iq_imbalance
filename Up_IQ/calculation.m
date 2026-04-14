clc; clear all;
I=1;
Q=1;
inv_I=0.715;
inv_Q=0.315;
e=(I*inv_I-Q*inv_Q-I*I+Q*Q)/(I*I+Q*Q);
p=(2*I*Q-I*inv_Q-inv_I*Q)/(I*I+Q*Q);
q=sin(0);



