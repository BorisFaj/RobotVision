function chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, modelo)
evaluados = [];
restantes = 1:Configuration.numObjects;
% Algunas funciones necesitan el tipo categorical para que no tome el
% verdadero/falso como continuo
y = categorical(objectsForTraining);
% Normalizar X para la RL
mu = mean(featuresForTraining);
sigma = std(featuresForTraining);
X=bsxfun(@minus, featuresForTraining, mu);
X=bsxfun(@rdivide, X, sigma);

%Nota: aun que se cambie el valor de la m dentro del for, en la siguiente
%iteracion no se va a tener el cuenta ese cambio
for m=1:Configuration.numObjects+1
    if(numel(restantes)<Configuration.numObjects)
        if(strcmp('NB',modelo))
            %Entrenar modelo
            NB = fitNaiveBayes(featuresForTraining,y(siguiente,:)','Distribution','kernel');            
            %Sacar las predicciones
            predictedNB = double(NB.predict(featuresForTest));
            predicted = predictedNB == 1;
        elseif(strcmp('RL',modelo))
            %Entrenar modelo
            RL = mnrfit(X,y(siguiente,:)');
            %Se predicen 2 columnas [falso positivo]
            predictedRL = mnrval(RL,featuresForTest);
            %Sacar las predicciones
            predicted = predictedRL(:,1) < 0.5;
        elseif(strcmp('RF',modelo))            
            %Entrenar modelo
            rng(1); % For reproducibility
            %Se calculan modelos con un numero de arboles de 1 a 50 con su
            %correspondiente out of the bag error
            RF = TreeBagger(50,featuresForTraining,y(siguiente,:)','OOBPred','On');   
            oobErr = oobError(RF);
            %Sacar las predicciones
            predicted = str2num(cell2mat(RF.predict(featuresForTest)));z
        end

        %Se obtienen las clases reales
        actual = objectsForTest(siguiente,:)';

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
            %                 TP FP TN FN Recall Precision FScore clasificador
            resultados(m-1,:) = [TP FP TN FN recall precision FScore siguiente];
        end      
        if(numel(restantes)>0)   
            [sinP resultadosH] = modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, restantes, modelo, true, true, false,X);        
        end
        %Se añade como caracteristica la prediccion realizada al TestSet
        featuresForTest = [featuresForTest predicted];

        %Tambien al training
        %Para añadir las nuevas predicciones al training, hay que clasificarlo
        if(strcmp('NB',modelo))
            predicted = double(NB.predict(featuresForTraining));    
            predicted = predicted == 1;
        elseif(strcmp('RL',modelo))
            predicted = mnrval(RL,featuresForTraining);
            %Si se predice todo negativo hay que añadir la columna de positivo
            %Convertirlo a logico
            predicted = predicted(:,1)<=0.5;
        elseif(strcmp('RF',modelo))
            predicted = str2num(cell2mat(RF.predict(featuresForTraining)));
        end

        featuresForTraining = [featuresForTraining predicted];
        %Se añade la ultima observacion (la del predictor) SIN NORMALIZAR
        X = [X featuresForTraining(:,size(featuresForTraining,2))];
    end
    
    if(numel(restantes)>0)           
        %Sacar el mejor FScore de los objetos restantes, teniendo en cuenta la
        %nueva prediccion      
        [conP resultadosC] = modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, restantes, modelo, true, true, false,X);        

        %Decidir el proximo objeto
        %[valor indice] = max(conP);
        if(numel(restantes)<Configuration.numObjects)
            [valor indice] = max((conP-sinP).*(conP+sinP));
        else
            [valor indice] = max(conP);
        end        
        siguiente = restantes(indice);

        %Actualizar listas de control
        restantes = [restantes(1:indice-1), restantes(indice+1:end)];
        evaluados = [evaluados siguiente];
    end       
end

%Apaño los resultados
%Añado en la ultima fila las medias
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
et = [cellstr('TP') cellstr('FP') cellstr('TN') cellstr('FN') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Objeto')];
resultados = num2cell(resultados);
resultados = [et;resultados]
end