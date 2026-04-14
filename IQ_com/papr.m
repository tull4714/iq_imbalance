clear all;

L = 100;
PAPR1 = [];
PAPR2 = [];

for i=1: L
    percent = i / L
    [papr1, papr2]= IQ_TR_PAPR(16, percent);
    PAPR1 = [PAPR1 papr1];
    PAPR2 = [PAPR2 papr2];
end
ccdf_plot(PAPR1,'-o');hold on;
ccdf_plot(PAPR2,'-*r');