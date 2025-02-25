function [v_bp,GP_bp,K_bp]=getBP(v_kph,GP,Ksplit)
%v_mesh=unique(v_kph);                        %Velocity breakpoints
v_bp=[0:0.5:100]';                            %Change v breakpoints for faster computation
GP_bp=unique(GP);                             %GP breakpoints
K_bp=unique(Ksplit);                          %Ksplit breakpoints
end