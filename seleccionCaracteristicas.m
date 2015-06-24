function nuevoDS = seleccionCaracteristicas(DS, objeto, algoritmo, iteraciones)
% Las caracteristicas elegidas utilizando weka con Cfs son:
% Objeto1: 3,31,33,57,66,69,85,125,239,257,258,260,261,262,263,264
%   -NB: 
%   -C45: 3,31,33,57,125,239,257,258,259
%   -RL: 
%   -SVM: 
% Objeto2: 1,4,8,14,60,70,213,233,255,257,258,259,260,263,264
%   -NB: 258,260,263,264
%   -C45: 1,4,8,60,213,233,255,257,258,259,260,263,264
%   -RL: 1,4,14,60,70,213,233,255,257,258,259,260,264
%   -SVM: 
% Objeto3: 2,3,4,7,14,16,35,72,117,160,186,192,204,242,254,255,257,258,259,260,262,263,264
%   -NB: 
%   -C45: 2,3,4,35,117,160,192,204,242,254,255,257,258,259,260,262,263,264
%   -RL: 
%   -SVM: 
% Objeto4: 2,74,199,255,257,258,262,263,264
%   -NB: 
%   -C45: 2,74,199,255,257,258,262,264
%   -RL: 2,74,263
%   -SVM: 
% Objeto5: 1,9,11,37,48,52,65,69,70,74,118,219,252,258,259,260,262,264
%   -NB: 
%   -C45: 
%   -RL: 
%   -SVM: 
% Objeto6: 2,3,8,12,167,259,261,262,263,264
%   -NB: 
%   -C45: 2,3,8,12,167,261,262,263,264
%   -RL: 
%   -SVM: 
% Objeto7: 2,3,4,5,7,14,17
%   -NB: 
%   -C45: 
%   -RL: 
%   -SVM: 
% Objeto8: 3,263
%   -NB: 
%   -C45: 
%   -RL: 
%   -SVM: 
% Pruebas terminadas, en los que no pone nada es porque el wrapper no
% descarta ninguno mas

if(objeto == 1)
    if(strcmp('C45',algoritmo))
        nuevoDS = DS(:,[3,31,33,57,125,239,257,258,259]);
    else
        nuevoDS = DS(:,[3,31,33,57,66,69,85,125,239,257,258,260,261,262,263,264]);
    end
elseif(objeto == 2)
    if(strcmp('NB',algoritmo))
        nuevoDS = DS(:,[258,260,263,264]);
    elseif(strcmp('C45',algoritmo))
        nuevoDS = DS(:,[1,4,8,60,213,233,255,257,258,259,260,263,264]);
    elseif(strcmp('RL',algoritmo))
        nuevoDS = DS(:,[1,4,14,60,70,213,233,255,257,258,259,260,264]);
    else
        nuevoDS = DS(:,[1,4,8,14,60,70,213,233,255,257,258,259,260,263,264]);
    end
elseif(objeto == 3)
    if(strcmp('C45',algoritmo))
        nuevoDS = DS(:,[2,3,4,35,117,160,192,204,242,254,255,257,258,259,260,262,263,264]);
    else
        nuevoDS = DS(:,[2,3,4,7,14,16,35,72,117,160,186,192,204,242,254,255,257,258,259,260,262,263,264]);
    end
elseif(objeto == 4)
    if(strcmp('C45',algoritmo))
        nuevoDS = DS(:,[2,74,199,255,257,258,262,264]);
    elseif(strcmp('RL',algoritmo))
        nuevoDS = DS(:,[2,74,263]);
    else
        nuevoDS = DS(:,[2,74,199,255,257,258,262,263,264]);
    end
elseif(objeto == 5)    
    nuevoDS = DS(:,[1,9,11,37,48,52,65,69,70,74,118,219,252,258,259,260,262,264]);
elseif(objeto == 6)     
    if(strcmp('C45',algoritmo))
        nuevoDS = DS(:,[2,3,8,12,167,261,262,263,264]);
    else
        nuevoDS = DS(:,[2,3,8,12,167,259,261,262,263,264]);
    end
elseif(objeto == 6)      
    if(strcmp('C45',algoritmo))
        nuevoDS = DS(:,[2,3,8,12,167,261,262,263,264]);
    else
        nuevoDS = DS(:,[2,3,8,12,167,259,261,262,263,264]);
    end    
elseif(objeto == 7)  
    nuevoDS = DS(:,[2,3,4,5,7,14,17]);
elseif(objeto == 8)  
    nuevoDS = DS(:,[3,263]);
end

if(iteraciones>0)
    nuevoDS = [nuevoDS,DS(:,265:(264+iteraciones))];
end
end