function buscarPatrones
    load datos/dbTrans.m;
       
    %Primero pruebo con la primera habitacion
    %Empiezo calculando el soporte de la habitacion con cada objeto por
    %separado
    A = [1,0,0,0,0,0,0,0,0,0];
    B = [0,1,0,0,0,0,0,0,0,0];
    AUB = A | B;
    
end

function sup(X)
    %El soporte se calcula con la formula de Kulczynski para que la medida
    %no se vea perjudicada por las transacciones nulas. Lo que se pretende
    %es buscar correlacion entre las variables. 
    %
    %           1/2* (s(AUB)/s(A) + s(AUB)/s(B))    
    for n=1:numel(X)
        if(X(n))
        end
    end
end