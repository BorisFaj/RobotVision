function extraerTransacciones(Configuration, objectsForTraining, clasesForTraining)  
    dbTrans = [];
    for n=1:Configuration.numClasses
        %Saca las transacciones para la habitacion n
        hab = objectsForTraining(:,(clasesForTraining == n)');
        transacciones = zeros(256,1);
        for h=1:size(hab,2)
            transacciones(bi2de(hab(:,h)')+1,:) = transacciones(bi2de(hab(:,h)')+1,:) +1;
        end
        %Quita las que son 0 y las mete en una tabla binaria
        %[b,b,b,b,b,b,b,b,n]
        hay = transacciones>0;
        ind = zeros(sum(hay),8);
        valor = zeros(sum(hay),1);
        cont = 1;
        for i=1:256
            if(hay(i))
                ind(cont,:) = dec2bin(uint8(i)-1,8);
                valor(cont,:) = transacciones(i,:);
                cont = cont+1;
            end
        end

        ind = ind-48;
        
        %Las mete en la base de datos de transacciones totales
        nhab = zeros(size(ind,1),1);
        nhab(:,:) = n;
        dbTrans = [dbTrans;nhab ind valor];
    end
    save medidas/dbTrans.mat dbTrans
end