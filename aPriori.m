function aPriori(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest)
    transacciones = zeros(256,1);
    for m=1:size(featuresForTraining,1)
        transacciones(bi2de(objectsForTraining(:,m)')+1,:) = transacciones(bi2de(objectsForTraining(:,m)')+1,:) +1;
    end
    hay = transacciones>0;
    ind = zeros(sum(hay),8);
    valor = zeros(sum(hay),1);
    cont = 1;
    for n=1:256
        if(hay(n))
            ind(cont,:) = dec2bin(uint8(n)-1,8);
            valor(cont,:) = transacciones(n,:);
            cont = cont+1;
        end
    end
    
    ind = ind-48;
    res = [ind valor]
    
    %Veces que aparece el objeto 1
    sum(res(res(:,1)>0,9))
end