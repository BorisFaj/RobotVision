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
evaluados = [];
restantes = 1:Configuration.numObjects;
y = categorical(objectsForTraining);
% Normalizar X para la RL
mu = mean(featuresForTraining);
sigma = std(featuresForTraining);
X=bsxfun(@minus, featuresForTraining, mu);
X=bsxfun(@rdivide, X, sigma);
%Falta decidir por que objeto se empieza
siguiente = 1;
restantes = [restantes(1:siguiente-1), restantes(siguiente+1:end)];
evaluados = [evaluados siguiente];
%Nota: aun que se cambie el valor de la m dentro del for, en la siguiente
%iteracion no se va a tener el cuenta ese cambio
for m=1:Configuration.numObjects
    m = siguiente;
       
    %Se entrenan los modelos
    NB = fitNaiveBayes(featuresForTraining,y(m,:)','Distribution','kernel');
    predictedNB = double(NB.predict(featuresForTest));

    RL = mnrfit(X,y(m,:)');
    %Se predicen 2 columnas [falso positivo]
    predictedRL = mnrval(RL,featuresForTest);
    if(size(predictedRL,2)==1)
        fprintf('Solo hay una claseee!!!\n');
        predictedRL = [predictedRL predictedRL == 0];
    end
    rng(1); % For reproducibility
    %Se calculan modelos con un numero de arboles de 1 a 50 con su
    %correspondiente out of the bag error
    RF = TreeBagger(50,featuresForTraining,y(m,:)','OOBPred','On');   
    oobErr = oobError(RF);
    
    %predicted = RF.predict(featuresForTest)
    
    %Se sacan las predicciones
    predicted = [predictedNB == 1, predictedRL(:,2) > 0.5, str2num(cell2mat(RF.predict(featuresForTest)))];
    %predicted = [predictedRL(:,2) str2num(cell2mat(RF.predict(featuresForTest)))];
    
    %Se obtienen las clases reales
    actual = objectsForTest(m,:)';

    %Se sacan los aciertos y los fallos
    for i=1:size(predicted,2) %modelo i
        %Para poder hacer las pruebas con el dataset pequeño hay que
        %inicializar a algo distinto de 0
        TP=0.01;TN=0.01;FP=0.01;FN=0.01;
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
    %Si todos eran NaN, saca el mejor TN
    if(isnan(valor))
        [valor indice] = max(resultados(:,3));
    end
    %Si todos eran 0, saca el mejor TP
    if(valor == 0)
        [valor indice] = max(resultados(:,1));
    end

    solucion = [solucion;resultados(indice,:)];
    

    %Para añadir las nuevas predicciones al training, hay que clasificarlo
    if(indice==1)
        nombre = 'NB';
        predictedN = double(NB.predict(featuresForTraining));    
        predictedN = predictedN == 1;
    elseif(indice==2)
        nombre = 'RL';
        predictedN = mnrval(RL,featuresForTraining);
        %Si se predice todo negativo hay que añadir la columna de positivo
        if(size(predictedN,2)==1)
            fprintf('Solo hay una claseee!!!\n');
            predictedN = [predictedN predictedN == 0];
        end
        %Convertirlo a logico
        predictedN = predictedN(:,2);
        predictedN = predictedN>=0.5;
    else
        nombre = 'RF';
        predictedN = str2num(cell2mat(RF.predict(featuresForTraining)));
    end
    
    %Sacar el mejor FScore de los objetos restantes, sin tener en cuenta la
    %nueva prediccion
    [sinP] = modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, restantes, nombre, true, true, false,X);
    
    % Se añade como caracteristica para el siguiente modelo (y siguiente
    % objeto) en training y en test
    featuresForTest = [featuresForTest predicted(:,indice)];
    featuresForTraining = [featuresForTraining predictedN];
    %Se añade la ultima observacion (la del predictor) SIN NORMALIZAR
    X = [X featuresForTraining(:,size(featuresForTraining,2))];
    
    %Sacar el mejor FScore de los objetos restantes, teniendo en cuenta la
    %nueva prediccion      
    [conP] = modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, restantes, nombre, true, true, false,X);
    
    %Decidir el proximo objeto
    [valor siguiente] = max(conP-sinP);
    
    %Actualizar listas de control
    restantes = [restantes(1:siguiente-1), restantes(siguiente+1:end)];
    evaluados = [evaluados siguiente];
   
    save solucionH.mat solucion;
end

%Apaño los resultados
%Añado en la ultima fila las medias
solucion(9,1:4) = sum(solucion(:,1:4));
solucion(9,5) = solucion(9,1)/(solucion(9,1)+solucion(9,4)); %recall
solucion(9,6) = solucion(9,1)/(solucion(9,1)+solucion(9,2)); %precision
solucion(9,7) = 2*solucion(9,6)*solucion(9,5)/(solucion(9,6)+solucion(9,5)); %FScore
%Y en cada objeto
for n=1:9
    %Los positivos
    total = solucion(n,1) + solucion(n,4);
    solucion(n,1) = solucion(n,1)/total;
    solucion(n,4) = solucion(n,4)/total;
    %Los negativos
    total = solucion(n,2) + solucion(n,3);
    solucion(n,2) = solucion(n,2)/total;
    solucion(n,3) = solucion(n,3)/total;
end
%Añado etiquetas
et = [cellstr('TP') cellstr('FP') cellstr('TN') cellstr('FN') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Clasificador') cellstr('Objeto')];
solucion = num2cell(solucion);
solucion = [et;solucion]
save solucionH.mat solucion;
end