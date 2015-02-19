function aPriori(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest)
    transacciones = zeros(256,1);
    for m=1:Configuration.numObjects
        transacciones(bi2de(objectsForTraining(:,m)')+1,:) = transacciones(bi2de(objectsForTraining(:,m)')+1,:) +1;
    end
end