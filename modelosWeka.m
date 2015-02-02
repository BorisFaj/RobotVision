function modelosWeka(Configuration, train, test, classindex, modelo, clasificar)
%
%   modelo: 'NB'|'RL'|'RF'
%    (Naive Bayes, Regresion Logistica, Random Forest)
%
%   clasificar: true|false
%    (si ademas de aprender los modelos hay que clasificar el training y
%    sacar las metricas)
%

fprintf(['Entrenando el clasificador ' modelo '\n']);
resultados = zeros(Configuration.numObjects+1,4);

for m=1:Configuration.numObjects
    %Train the classifier
    if(strcmp('NB',modelo))
        clasificador = trainWekaClassifier(train(m),'bayes.NaiveBayes');
    elseif(strcmp('RL',modelo))
        clasificador = trainWekaClassifier(train(m),'functions.Logistic');
    elseif(strcmp('RF',modelo))
        clasificador = trainWekaClassifier(train(m),'trees.RandomForest');
    else
        error('ERROR! El parametro modelo de la funcion modelosWeka solo admite las cadenas: NB|RL|RF (Naive Bayes, Regresion Logistica, Random Forest)');
    end

    save(strcat(modelo,'.mat'), 'clasificador');
    
    if(clasificar)
        %Test the classifier
        predicted = wekaClassify(test(m),clasificador);

        %The actual class labels (i.e. indices thereof)
        actual = test.attributeToDoubleArray(classindex-1); %java indexes from 0

        errorRate = sum(actual ~= predicted)/500

        TP=0;TN=0;FP=0;FN=0;

        for n=1:length(predicted)
            if(predicted(n)) %se predice positivo
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
        resultados(m,1) = TP/(TP+FN); %recall
        resultados(m,2) = TP/(TP+FP); %precision
        resultados(m,3) = 2*resultados(m,2)*resultados(m,1)/(resultados(m,2)+resultados(m,1)); %FScore
        resultados(m,4) = errorRate; %error rate

        resultados(Configuration.numObjects+1, 1) =resultados(Configuration.numObjects+1, 1)+resultados(m,1);
        resultados(Configuration.numObjects+1, 2) =resultados(Configuration.numObjects+1, 2)+resultados(m,2);
        resultados(Configuration.numObjects+1, 3) =resultados(Configuration.numObjects+1, 3)+resultados(m,3);
        resultados(Configuration.numObjects+1, 4) =resultados(Configuration.numObjects+1, 4)+resultados(m,4);
    end
end

if(clasificar)
    %Apañar la matriz de resultados
    resultados(Configuration.numObjects+1, 1) =resultados(Configuration.numObjects+1, 1)/Configuration.numObjects;
    resultados(Configuration.numObjects+1, 2) =resultados(Configuration.numObjects+1, 2)/Configuration.numObjects;
    resultados(Configuration.numObjects+1, 3) =resultados(Configuration.numObjects+1, 3)/Configuration.numObjects;
    resultados(Configuration.numObjects+1, 4) =resultados(Configuration.numObjects+1, 4)/Configuration.numObjects;

    resultados = num2cell(resultados);
    obj = [cellstr('Obj1');cellstr('Obj2');cellstr('Obj3');cellstr('Obj4');cellstr('Obj5');cellstr('Obj6');cellstr('Obj7');cellstr('Obj8');cellstr('Media')];
    et = [cellstr('-') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Error Rate')];

    resultados = [obj resultados];
    resultados = [et;resultados]

    save(strcat('resultados',modelo,'.mat'), 'resultados');
end
end