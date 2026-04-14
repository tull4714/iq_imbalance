function[out_vector]=hard_decision(in_vector,M) %in_vector은 수신신호
%===============================================%
% 1. function  : hard decision을 수행
% 2. argument  : input     in_vector : 입력벡터
%                           M : m-ary
%                    output  put-vector : 출력벡터
%==============================================%
if M==2
    out_vector= sign(real(in_vector));
elseif M==4                       %SER
    r_part=real(in_vector);
    i_part=imag(in_vector);
    out_vector=sign(r_part)+j*sign(i_part);
end;
