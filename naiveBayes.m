function naiveBayes(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest)

X = featuresForTraining;
y = objectsForTraining;
%por que los valores son tan proximos a 0?? es que esta normalizado?¿?
for m = 1:Configuration.numObjects
    modelos{m} = {fitNaiveBayes(X,y(m,:))};
end

clasificar(Configuration, modelos, featuresForTest, objectsForTest)

end

function clasificar(Configuration,modelos, X, y)
    TP=0;TN=0;FP=0;FN=0;sin=0;
    resultados = zeros(Configuration.numObjects+1,8);
    % Evaluar resultados    
    for n = 1:length(y)
        for m = 1:Configuration.numObjects        
            mo = modelos{1,m};
            h = predict(mo{1,1},X(n,:));
            if(h==1)   % se clasifica positivo
                if(y(m,n) == 1) % es positivo
                    resultados(m,1)=resultados(m,1)+1; %TP
                else
                    resultados(m,2)=resultados(m,2)+1; %FP
                end
            else           % se clasifica negativo
                if(y(m,n) == 0) % es negativo
                    resultados(m,3)=resultados(m,3)+1; %TN
                else
                    resultados(m,4)=resultados(m,4)+1; %FN
                end  
            end
        end
    end
    
    % Metricas [TP, TN, FP, FN, precision, recall, F1Score]
    for n=1:Configuration.numObjects
        % Se calculan las que faltan
        resultados(n,5) = resultados(n,1)/(resultados(n,1)+resultados(n,4)); %recall = TP/(TP+FN)
        resultados(n,6) = resultados(n,1)/(resultados(n,1)+resultados(n,3)); %precision = TP/(TP+FP)
        resultados(n,7) = 2*resultados(n,6)*resultados(n,5)/(resultados(n,5)+resultados(n,6)); %FScore = 2*precision*recall/(precision+recall)
        resultados(n,8) = (resultados(n,1)+resultados(n,2))/(resultados(n,1)+resultados(n,4)+resultados(n,2)+resultados(n,3)); %accuracy = TP+TN/P+N
        % Se normalizan las metricas que ya hay
        totalP = resultados(n,1)+resultados(n,4); %TP+FN
        totalN =  resultados(n,2)+resultados(n,3); %TN+FP
        resultados(n,1) = resultados(n,1)/totalP; %TP
        resultados(n,2) = resultados(n,2)/totalN; %TN
        resultados(n,3) = resultados(n,3)/totalN; %FP
        resultados(n,4) = resultados(n,4)/totalP; %FN
    end   
    
    % Media total
    for n=1:8
        resultados(Configuration.numObjects+1,n) = sum(resultados(:,n))/Configuration.numObjects;
    end
    resultados = num2cell(resultados);
    etiquetasX = [cellstr('TP%'),cellstr('TN%'),cellstr('FP%'),cellstr('FN%'),cellstr('recall'),cellstr('precision'),cellstr('FScore'),cellstr('Accuracy')];
    etiquetasY = [cellstr('-');cellstr('Ojb1');cellstr('Ojb2');cellstr('Ojb3');cellstr('Ojb4');cellstr('Ojb5');cellstr('Ojb6');cellstr('Ojb7');cellstr('Ojb8');cellstr('Media')];
    resultados = [etiquetasX;resultados];
    resultados = [etiquetasY,resultados];
    resultados = [resultados(:,1:3),resultados(:,7),resultados(:,6),resultados(:,8:9)];
    save resultadosNB.mat resultados    
end
