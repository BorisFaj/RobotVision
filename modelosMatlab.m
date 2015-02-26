function [FScores, resultados] = modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, objetosEvaluar, modelo, entrenar, clasificar, guardar,X)
%   Aviso: Si NO se entena y SI se clasifica, no va bien porque no cuenta los TP, etc
%
%   modelo: 'NB'|'RL'|'RF'|'DT'|'SVM'
%    (Naive Bayes, Regresion Logistica, Random Forest, Decision Tree, SVM)
%
%   clasificar: true|false
%    (si ademas de aprender los modelos hay que clasificar el training y
%    sacar las metricas)
%
resultados = zeros(numel(objetosEvaluar)+1,8);
if(entrenar)
    fprintf(['Entrenando el clasificador ' modelo '\n']);
y = categorical(objectsForTraining);
    for m=1:numel(objetosEvaluar)
        indice = objetosEvaluar(m);
        %Train the classifier
        if(strcmp('NB',modelo))
            clasificador = fitNaiveBayes(featuresForTraining,y(indice,:)', 'Distribution','kernel');
        elseif(strcmp('RL',modelo))
            if(X==0)
                mu = mean(featuresForTraining);
                sigma = std(featuresForTraining);
                X=bsxfun(@minus, featuresForTraining, mu);
                X=bsxfun(@rdivide, X, sigma);  
            end
                %clasificador = mnrfit(X,y(indice,:)');
                clasificador = mnrfit(X,y(indice,:)','model','nominal');
        elseif(strcmp('RF',modelo))
            rng(1);
            clasificador = TreeBagger(50,featuresForTraining,y(indice,:)','OOBPred','On');
        elseif(strcmp('DT',modelo))
            clasificador = fitctree(X,y(indice,:)');    
        elseif(strcmp('SVM',modelo))
            clasificador = fitcsvm(X,y(indice,:)');            
        else
            error('ERROR! El parametro modelo de la funcion modelosWeka solo admite las cadenas: NB|RL|RF|DT|SVM (Naive Bayes, Regresion Logistica, Random Forest, Decision Tree)');
        end

        if(clasificar)
            if(strcmp('NB',modelo))
                predicted = (double(clasificador.predict(featuresForTest))-1);
                %Puede generar valores NaN, los tomo como negativos
                predicted = predicted == 1;
            elseif(strcmp('RL',modelo))
                predicted = mnrval(clasificador,featuresForTest);
                predicted = predicted(:,1);
                predicted = predicted<=0.5;
            elseif(strcmp('RF',modelo))
                predicted = str2num(cell2mat(clasificador.predict(featuresForTest)));        
            elseif(strcmp('DT',modelo))
                predicted = clasificador.predict(featuresForTest);    
                predicted = predicted == categorical(1);
            elseif(strcmp('SVM',modelo))
                [label,score] = predict(clasificador,featuresForTest);
                predicted = label == categorical(1);                 
            else
                error('ERROR! El parametro modelo de la funcion modelosWeka solo admite las cadenas: NB|RL|RF|DT|SVM (Naive Bayes, Regresion Logistica, Random Forest, Decision Tree)');
            end

            %The actual class labels (i.e. indices thereof)
            actual = objectsForTest(indice,:)';

            TP=0.001;TN=0.001;FP=0.001;FN=0.001;
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
            resultados(m,4) = (TP+TN)/(TP+TN+FP+FN); %accuracy
            resultados(m,5) = TP; %TP
            resultados(m,6) = TN; %TN
            resultados(m,7) = FP; %FP
            resultados(m,8) = FN; %FN
        end
    end
else
    load(strcat('medidas/modelo',modelo,'Matlab.mat'));
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
    resultados(numel(objetosEvaluar)+1, 5) = TP/(TP+FN);
    resultados(numel(objetosEvaluar)+1, 6) = TN/(TN+FP);
    resultados(numel(objetosEvaluar)+1, 7) = FP/(TN+FP);
    resultados(numel(objetosEvaluar)+1, 8) = FN/(TP+FN);
    
    resultados(numel(objetosEvaluar)+1, 1) = TP/(TP+FN);
    resultados(numel(objetosEvaluar)+1, 2) = TP/(TP+FP);
    resultados(numel(objetosEvaluar)+1, 3) = 2*resultados(numel(objetosEvaluar)+1, 2)*resultados(numel(objetosEvaluar)+1, 1)/(resultados(numel(objetosEvaluar)+1, 2)+resultados(numel(objetosEvaluar)+1, 1));
    resultados(numel(objetosEvaluar)+1, 4) = (TP+TN)/total;
    
    %Calcular las tasas locales de TP, TN, etc (machacando los valores
    %absolutos de cada objeto)
    for n=1:numel(objetosEvaluar)
        totalP = sum(resultados(n, [5 8]));
        totalN = sum(resultados(n, [6 7]));
        resultados(n,5) = resultados(n,5)/totalP; %TP
        resultados(n,6) = resultados(n,6)/totalN; %TN
        resultados(n,7) = resultados(n,7)/totalN; %FP
        resultados(n,8) = resultados(n,8)/totalP; %FN
    end
    
    %Vector que devuelve la funcion
    FScores = resultados(1:numel(objetosEvaluar),3);
    
    %Poner las etiquetas
    resultados = num2cell(resultados);
    obj = cellstr('Obj1');
    for o=2:numel(objetosEvaluar)
        obj = [obj; cellstr(strcat('Obj',num2str(o)))];
    end
    obj = [obj;cellstr('Total')];
    et = [cellstr('-') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Accuracy') cellstr('TP%') cellstr('TN%') cellstr('FP%') cellstr('FN%')];

    resultados = [obj resultados];
    resultados = [et;resultados];
    if(guardar)
        save(strcat('medidas/resultados',modelo,'Matlab.mat'), 'resultados');
    end
end
end