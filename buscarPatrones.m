function buscarPatrones
    clear;
    load medidas/dbTrans.mat;
    %La confianza se calcula con la formula de Kulczynski para que la medida
    %no se vea perjudicada por las transacciones nulas. Lo que se pretende
    %es buscar correlacion entre las variables. 
    %
    %           1/2* (s(AUB)/s(A) + s(AUB)/s(B))  
       
    % En el problema actual se pretende sacar las relaciones de las
    % habitaciones con los objetos, por lo que se ha modificado la formula
    % añadiendo el grupo de objetos C, que solo es la habitación
    %           
    %           1/3* (s(AUBUC)/s(A) + s(AUBUC)/s(B) + s(AUBUC)/s(C))  
    
    % Las permutaciones se van a generar a partir de los numeros generados
    % con 2⁸ bits
    combinaciones = zeros(8,8);
    for n=1:8
        combinaciones(n,n) = 1;
    end
    resultados = zeros(1,10);
    % Se calculan los soportes de conjuntos de tamaño 2
    % Se prueba desde la habitacion 0, es decir, con independencia de la
    % habitacion
    for h=1:10
        hab = zeros(8,1);
        A = [hab combinaciones];
        % Para cada objeto
        for ob=1:8
            obActual = zeros(1,9);
            obActual(ob+1) = 1;
            %obActual(1)=h;
            % Para cada combinacion
            for n=1:8
                % Si no es la combinacion del objeto consigo mismo
                if(sum(A(n,2:end) == obActual(2:end)) ~= 8)
                    AUB = [h A(n,2:end) | obActual(2:end)];
                    sA = sup(A(n,:),dbTrans);
                    sB = sup(obActual,dbTrans);
                    sC = sup([h zeros(1,8)],dbTrans);
                    sAB = sup(AUB,dbTrans);
                    resultados(size(resultados,1)+1,:) = [AUB 1/3*(sAB/sA + sAB/sB + sAB/sC)];
                end
            end
        end    
    end
    %Limpio los NaN
    resultados(isnan(resultados(:,10)),10) = 0;
    %Limpio las que son 0
    resultados = resultados(resultados(:,10)>0,:);
    save medidas/soportes.mat resultados
end

function [sup] = sup(X,dbTrans)
    sup = ones(size(dbTrans,1),1);
    for n=1:numel(X)
        if(X(n)~=0)
            sup = (X(n) == dbTrans(:,n)) & sup;
        end        
    end
    %Hay que tener en cuenta que cada vez que aparece una
    %transaccion en dbTrans tiene asociado un numero de veces que
    %ha aparecido en la bd original, no tiene por que ser 1!!
    sup = sum(dbTrans(sup,10));
end