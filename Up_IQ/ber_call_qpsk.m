function [ber]=ber_call_qpsk(A, B, M) %A는 하드디시젼 신호, B는 원신호
%----------------------------------------%
% 1. function : Bit Error Rate을 계산
% 2. argument :   input   A: 비교 벡터
%                         B: 비교대상 벡터
%                output   out : BER
%=======================================%
numoferr=0;
%초기값 설정
NumOfBitError =0 ;
NumOfSymbolError=0;
SymbolError=0;
integral =0;
%%%%%%%%%%%%%%%%%%%%%%%%
D= size(A,1)*size(A,2);
if M==2
    ber = sum(~(B==A))/D;
    %%%%%%%%%%%%%%%%%%%%%
elseif M==4
    L=length(A);
    for q=1:L,
        integral_I=real(A(q));
        integral_Q=imag(A(q));
        B_I=real(B(q));
        B_Q=imag(B(q));
        % I-Channel decision
        if(integral_I>0)
            decision_I=1;
        else
            decision_I=-1;
        end
        % Q-Channel decision
        if(integral_Q>0)
            decision_Q=1;
        else
            decision_Q=-1;
        end
        % increment the error counter
        if(decision_I ~=B_I)   %i-channel
            NumOfBitError =NumOfBitError +1;  %Bit Error Counter
            SymbolError = 1;                  %Symbol Error
        end
        if(decision_Q ~=B_Q)    %Q-channel
            NumOfBitError=NumOfBitError +1;   %Bit Error Counter
            SymbolError= 1;                   %Symbol Error
        end
        if(SymbolError ==1)
            NumOfSymbolError= NumOfSymbolError+1; %Symbol Error counter 
        end
        SymbolError = 0;
    end
    pb = NumOfBitError/(2*L) ;  %Bit Error Rate
    ps = NumOfSymbolError/L ;   %Symbol Error Rate
    ber=pb;
end;
