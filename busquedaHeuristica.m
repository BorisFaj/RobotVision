function busquedaHeuristica(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest)
% Se van a entrenar 3 modelos por cada objeto y nos vamos a quedar con el
% que mejor FScore tenga, a partir de ese vamos a entrenar el siguiente
% objeto y a proceder de la misma manera. El resultado va a estar
% condicionado al objeto por el que se empiece, por lo que se va a repetir
% el proceso una vez por objeto

% *IDEA: probar algo parecido a random forest. Elegir cada vez un primer
% objeto diferente y el siguiente objeto se escoge aleatoriamente en lugar
% de secuencialmente.

solucion = [];
aleatorio = [1:8];
aleatorio = aleatorio(randperm(length(aleatorio))); 

%Nota: aun que se cambie el valor de la m dentro del for, en la siguiente
%iteracion no se va a tener el cuenta ese cambio
for m=1:Configuration.numObjects
    m = aleatorio(m);
    %Parche al error producido porque con el dataset pequeño, no hay
    %objetos de la clase 5 ni 7
    if(m==5)
        m=6;
    elseif(m==7)
        m=8;
    end
    
    
    [train, test] = convertirAWeka(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, true);
    
    %Se entrenan los modelos
    NB = trainWekaClassifier(train(m),'bayes.NaiveBayes');
    RL = trainWekaClassifier(train(m),'functions.Logistic');
    RF = trainWekaClassifier(train(m),'trees.RandomForest');
    
    %Se sacan las predicciones
    predicted = [wekaClassify(test(m),NB) wekaClassify(test(m),RL) wekaClassify(test(m),RF)];
    
    %Se obtienen las clases reales
    actual = test(m).attributeToDoubleArray(size(featuresForTraining,2));
    
    errorRate = [sum(actual ~= predicted(:,1)) sum(actual ~= predicted(:,2)) sum(actual ~= predicted(:,3))]/size(predicted,1);
    [m errorRate]

    %Se sacan los aciertos y los fallos
    for i=1:size(predicted,2) %modelo i
        TP=0;TN=0;FP=0;FN=0;
        for n=1:size(predicted,1) %ejemplo n
            if(predicted(n,i)) %se predice positivo
                if(actual(n))    %es positivo
                    TP=TP+1;
                else
                    FP = FP+1;
                end
            else            %se predice negativo
                if(~actual(n))  %es negativo
                    TN=TN+1;
                else
                    FN=FN+1;
                end
            end
        end        
        recall = (TP/(TP+FN));
        precision = (TP/(TP+FP));
        FScore = 2*precision*recall/(precision+recall);
        %                 TP FP TN FN Recall Precision FScore clasificador
        %                 objeto
        resultados(i,:) = [TP FP TN FN recall precision FScore i m];
    end
    
    % Se saca el clasificador con mejor FScore
    [valor indice] = max(resultados(:,7));
    solucion = [solucion;resultados(indice,:)];
    
    % Se añade como caracteristica para el siguiente modelo (y siguiente
    % objeto) en training y en test
    featuresForTest = [featuresForTest predicted(:,indice)];
    %Para añadirlas al training hay que clasificarlo
    if(indice==1)
        predicted = wekaClassify(train(m),NB);
    elseif(indice==2)
        predicted = wekaClassify(train(m),RL);
    else
        predicted = wekaClassify(train(m),RF);
    end

    featuresForTraining = [featuresForTraining predicted];
   
    save solucionH.mat solucion;
end

%Apaño los resultados
%Añado en la ultima fila las medias
solucion(9,1:4) = sum(solucion(:,1:4));
solucion(9,5) = solucion(9,1)/(solucion(9,1)+solucion(9,4)); %recall
solucion(9,6) = solucion(9,1)/(solucion(9,1)+solucion(9,2)); %precision
solucion(9,7) = 2*solucion(9,6)*solucion(9,5)/(solucion(9,6)+solucion(9,5)); %FScore
%Paso TP FP TN y FN a % en el total
total = sum(sum(solucion(:,1:4)));
solucion(9,1:4)=solucion(9,1:4)/total
%Y en cada objeto
total = sum(solucion(m,1:4));
solucion(1:8,1:4) = solucion(1:8,1:4)/total;
%Añado etiquetas
et = [cellstr('TP') cellstr('FP') cellstr('TN') cellstr('FN') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Clasificador') cellstr('Objeto')];
solucion = num2cell(solucion);
solucion = [et;solucion]
save solucionH.mat solucion;
end