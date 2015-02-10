function modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, modelo, entrenar, clasificar)
%   Aviso: Si NO se entena y SI se clasifica, no va bien porque no cuenta los TP, etc
%
%   modelo: 'NB'|'RL'|'RF'
%    (Naive Bayes, Regresion Logistica, Random Forest)
%
%   clasificar: true|false
%    (si ademas de aprender los modelos hay que clasificar el training y
%    sacar las metricas)
%
resultados = zeros(Configuration.numObjects+1,8);
if(entrenar)
    fprintf(['Entrenando el clasificador ' modelo '\n']);
y = categorical(objectsForTraining);
    for m=1:Configuration.numObjects
        %Train the classifier
        if(strcmp('NB',modelo))
            clasificador = fitNaiveBayes(featuresForTraining,y(m,:)');
        elseif(strcmp('RL',modelo))
                mu = mean(featuresForTraining);
                sigma = std(featuresForTraining);
                X=bsxfun(@minus, featuresForTraining, mu);
                X=bsxfun(@rdivide, X, sigma);    
                clasificador = mnrfit(X,y(m,:)');
        elseif(strcmp('RF',modelo))
            rng(1);
            clasificador = TreeBagger(50,featuresForTraining,y(m,:)','OOBPred','On');
        else
            error('ERROR! El parametro modelo de la funcion modelosWeka solo admite las cadenas: NB|RL|RF (Naive Bayes, Regresion Logistica, Random Forest)');
        end

        save(strcat('modelo',modelo,'Matlab.mat'), 'clasificador');

        if(clasificar)
            if(strcmp('NB',modelo))
                predicted = (double(clasificador.predict(featuresForTest))-1);
            elseif(strcmp('RL',modelo))
                predictedRL = mnrval(clasificador,featuresForTest);
                predicted = predictedRL(:,2);
                predicted = predicted>=0.5;
            elseif(strcmp('RF',modelo))
                predicted = str2num(cell2mat(clasificador.predict(featuresForTest)));                
            else
                error('ERROR! El parametro modelo de la funcion modelosWeka solo admite las cadenas: NB|RL|RF (Naive Bayes, Regresion Logistica, Random Forest)');
            end

            %The actual class labels (i.e. indices thereof)
            actual = objectsForTest(m,:)';

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
            resultados(m,4) = NaN; %error rate
            resultados(m,5) = TP; %TP
            resultados(m,6) = TN; %TN
            resultados(m,7) = FP; %FP
            resultados(m,8) = FN; %FN
        end
    end
else
    load(strcat('modelo',modelo,'Matlab.mat'));
end
%Si NO se entena y SI se clasifica, no va bien porque no cuenta los TP, etc
if(clasificar)
    %Apañar la matriz de resultados
    %Recall, precision y FScore se calculan con los datos totales
    TP = sum(resultados(:,5));
    TN = sum(resultados(:,6));
    FP = sum(resultados(:,7));
    FN = sum(resultados(:,8));
    total = sum(sum(resultados(:, [5 6 7 8])));
    resultados(Configuration.numObjects+1, 5) = TP/(TP+FN)
    resultados(Configuration.numObjects+1, 6) = TN/(TN+FP)
    resultados(Configuration.numObjects+1, 7) = FP/(TN+FP)
    resultados(Configuration.numObjects+1, 8) = FN/(TP+FN)
    
    resultados(Configuration.numObjects+1, 1) = TP/(TP+FN);
    resultados(Configuration.numObjects+1, 2) = TP/(TP+FP);
    resultados(Configuration.numObjects+1, 3) = 2*resultados(Configuration.numObjects+1, 2)*resultados(Configuration.numObjects+1, 1)/(resultados(Configuration.numObjects+1, 2)+resultados(Configuration.numObjects+1, 1));
    resultados(Configuration.numObjects+1, 4) = 1-((TP+TN)/total);
    %Calcular las tasas totales de TP, TN, etc
    for n=(size(resultados,2)-3):size(resultados,2)
        resultados(Configuration.numObjects+1, n) =(sum(resultados(:, n))/total)*100;
    end
    
    %Calcular las tasas locales de TP, TN, etc (machacando los valores
    %absolutos de cada objeto)
    for n=1:Configuration.numObjects
        totalP = sum(resultados(n, [5 8]));
        totalN = sum(resultados(n, [6 7]));
        resultados(n,5) = resultados(n,5)/totalP; %TP
        resultados(n,6) = resultados(n,6)/totalN; %TN
        resultados(n,7) = resultados(n,7)/totalN; %FP
        resultados(n,8) = resultados(n,8)/totalP; %FN
    end

    %Poner las etiquetas
    resultados = num2cell(resultados);
    obj = [cellstr('Obj1');cellstr('Obj2');cellstr('Obj3');cellstr('Obj4');cellstr('Obj5');cellstr('Obj6');cellstr('Obj7');cellstr('Obj8');cellstr('Total')];
    et = [cellstr('-') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Error Rate') cellstr('TP%') cellstr('TN%') cellstr('FP%') cellstr('FN%')];

    resultados = [obj resultados];
    resultados = [et;resultados]

    save(strcat('resultados',modelo,'Matlab.mat'), 'resultados');
end
end