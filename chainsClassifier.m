function chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, modelo, orden)
% Algunas funciones necesitan el tipo categorical para que no tome el
% verdadero/falso como continuo
y = categorical(objectsForTraining);

%Nota: aun que se cambie el valor de la m dentro del for, en la siguiente
%iteracion no se va a tener el cuenta ese cambio
for m=1:Configuration.numObjects
    if(strcmp('NB',modelo))
        %FSS
        reducidoTr = seleccionCaracteristicas(featuresForTraining, orden(m), 'NB',m-1);
        reducidoTe = seleccionCaracteristicas(featuresForTest, orden(m), 'NB',m-1);
        %Entrenar modelo
        NB = fitNaiveBayes(reducidoTr,y(orden(m),:)','Distribution','kernel');            
        %Sacar las predicciones
        predictedNB = (double(NB.predict(reducidoTe))-1);
        predicted = predictedNB == 1;
    elseif(strcmp('RL',modelo))
        %FSS
        reducidoTr = seleccionCaracteristicas(featuresForTraining, orden(m), 'RL',m-1);
        reducidoTe = seleccionCaracteristicas(featuresForTest, orden(m), 'RL',m-1);
        %Normaliza
        mu = mean(reducidoTr);
        sigma = std(reducidoTr);
        X=bsxfun(@minus, reducidoTr, mu);
        X=bsxfun(@rdivide, X, sigma);
        %Entrenar modelo
        RL = mnrfit(X,y(orden(m),:)');
        %Se predicen 2 columnas [falso positivo]
        predictedRL = mnrval(RL,reducidoTe);
        %Sacar las predicciones
        predicted = predictedRL(:,1) < 0.5;
    elseif(strcmp('RF',modelo))            
        %Entrenar modelo
        rng(1); % For reproducibility
        %Se calculan modelos con un numero de arboles de 1 a 50 con su
        %correspondiente out of the bag error
        RF = TreeBagger(50,featuresForTraining,y(orden(m),:)','OOBPred','On');   
        oobErr = oobError(RF);
        %Sacar las predicciones
        predicted = str2num(cell2mat(RF.predict(featuresForTest)));
    elseif(strcmp('DT',modelo))
        %FSS
        reducidoTr = seleccionCaracteristicas(featuresForTraining, orden(m), 'C45',m-1);
        reducidoTe = seleccionCaracteristicas(featuresForTest, orden(m), 'C45',m-1);

        DT = fitctree(reducidoTr,y(orden(m),:)','SplitCriterion','deviance');
        predicted = DT.predict(reducidoTe);
        predicted = predicted == categorical(1);
    elseif(strcmp('SVM',modelo))
        %FSS
        reducidoTr = featuresForTraining;
        reducidoTe = featuresForTest;
        [~, SVM, predicted] = SVMObjetos(Configuration, reducidoTr, reducidoTe, objectsForTraining(orden(m),:), objectsForTest(orden(m),:));     
        predicted = predicted';
    end

    %Se obtienen las clases reales
    actual = objectsForTest(orden(m),:)';

    %Se sacan los aciertos y los fallos
    TP=0;TN=0;FP=0;FN=0;
    for i=1:length(predicted) %ejemplo i
        %Para poder hacer las pruebas con el dataset pequeño hay que
        %inicializar a algo distinto de 0            
        if(predicted(i)) %se predice positivo
            if(actual(i))    %es positivo
                TP=TP+1;
            else
                FP = FP+1;
            end
        else            %se predice negativo
            if(~actual(i))  %es negativo
                TN=TN+1;
            else
                FN=FN+1;
            end
        end        
        recall = (TP/(TP+FN));
        precision = (TP/(TP+FP));
        FScore = 2*precision*recall/(precision+recall);
        accuracy = (TP+TN)/(TP+TN+FP+FN);
        %                 TP FP TN FN Recall Precision FScore clasificador
        resultados(m,:) = [TP FP TN FN recall precision FScore accuracy orden(m)];
    end      

    %Se añade como caracteristica la prediccion realizada al TestSet
    featuresForTest = [featuresForTest predicted];

    %Tambien al training
    %Para añadir las nuevas predicciones al training, hay que clasificarlo
    if(strcmp('NB',modelo))
        predicted = double(NB.predict(reducidoTr));    
        predicted = predicted == 1;
    elseif(strcmp('RL',modelo))
        predicted = mnrval(RL,reducidoTr);
        %Si se predice todo negativo hay que añadir la columna de positivo
        %Convertirlo a logico
        predicted = predicted(:,1)<=0.5;
    elseif(strcmp('RF',modelo))
        predicted = str2num(cell2mat(RF.predict(reducidoTr)));
    elseif(strcmp('DT',modelo))
        predicted = DT.predict(reducidoTr);    
        predicted = predicted == categorical(1);
    elseif(strcmp('SVM',modelo))      
        %Predecir SVM
        [predicted,~]=model_predict(reducidoTr',SVM,0);
        for n=1:numel(predicted)
            if(predicted(n)==-1)
                predicted(n)=0;
            else
                predicted(n)=1;
            end
        end
        predicted = predicted';
    end

    featuresForTraining = [featuresForTraining predicted];
    %Se añade la ultima observacion (la del predictor) SIN NORMALIZAR
    %X = [X featuresForTraining(:,size(featuresForTraining,2))];
   
        
end

%Apaño los resultados
%Añado en la ultima fila las medias
resultados(9,8) = (sum(resultados(:,1))+sum(resultados(:,3)))/sum(sum(resultados(:,1:4))); %Accuracy medio
resultados(9,1:4) = sum(resultados(:,1:4));
resultados(9,5) = resultados(9,1)/(resultados(9,1)+resultados(9,4)); %recall
resultados(9,6) = resultados(9,1)/(resultados(9,1)+resultados(9,2)); %precision
resultados(9,7) = 2*resultados(9,6)*resultados(9,5)/(resultados(9,6)+resultados(9,5)); %FScore
%Y en cada objeto
for n=1:9
    %Los positivos
    total = resultados(n,1) + resultados(n,4);
    resultados(n,1) = resultados(n,1)/total;
    resultados(n,4) = resultados(n,4)/total;
    %Los negativos
    total = resultados(n,2) + resultados(n,3);
    resultados(n,2) = resultados(n,2)/total;
    resultados(n,3) = resultados(n,3)/total;
end
%Añado etiquetas
et = [cellstr('TP') cellstr('FP') cellstr('TN') cellstr('FN') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Accuracy') cellstr('Objeto')];
resultados = num2cell(resultados);
resultados = [et;resultados]

save(strcat('medidas/Chains',modelo,'.mat'), 'resultados');
save(strcat('datos/prediccionesTraining',modelo,'.mat'), 'featuresForTraining');
save(strcat('datos/prediccionesTest',modelo,'.mat'), 'featuresForTest');
end