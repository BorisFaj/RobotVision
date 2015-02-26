function clasificaHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, modelo)
%
%   modelo: 'NB'|'RL'|'RF'|'DT'|'SVM'
%    (Naive Bayes, Regresion Logistica, Random Forest, Decision Tree, SVM)
%

% Normalizar X para la RL
mu = mean(featuresForTraining);
sigma = std(featuresForTraining);
X=bsxfun(@minus, featuresForTraining, mu);
X=bsxfun(@rdivide, X, sigma);

%Nota: aun que se cambie el valor de la m dentro del for, en la siguiente
%iteracion no se va a tener el cuenta ese cambio
for m=1:Configuration.numClases
    if(strcmp('NB',modelo))
        %Entrenar modelo
        NB = fitNaiveBayes(featuresForTraining,clasesForTraining==m,'Distribution','kernel');            
        %Sacar las predicciones
        predictedNB = double(NB.predict(featuresForTest));
        predicted = predictedNB == 1;
    elseif(strcmp('RL',modelo))
        %Entrenar modelo
        RL = mnrfit(X,clasesForTraining==m);
        %Se predicen 2 columnas [falso positivo]
        predictedRL = mnrval(RL,featuresForTest);
        %Sacar las predicciones
        predicted = predictedRL(:,1) < 0.5;
    elseif(strcmp('RF',modelo))            
        %Entrenar modelo
        rng(1); % For reproducibility
        %Se calculan modelos con un numero de arboles de 1 a 50 con su
        %correspondiente out of the bag error
        RF = TreeBagger(50,featuresForTraining,clasesForTraining==m,'OOBPred','On');   
        oobErr = oobError(RF);
        %Sacar las predicciones
        predicted = str2num(cell2mat(RF.predict(featuresForTest)));
	elseif(strcmp('DT',modelo))
        DT = fitctree(X,clasesForTraining==m);
        predicted = DT.predict(featuresForTest);
        predicted = predicted == categorical(1);
	elseif(strcmp('SVM',modelo))
        SVM = fitcsvm(X,clasesForTraining==m);
        [label,score] = predict(SVM,featuresForTest);
        predicted = label == categorical(1);                 
    end

    %Se obtienen las clases reales
	actual = clasesForTest==m;

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
	resultados(m,:) = [TP FP TN FN recall precision FScore];
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

save(strcat('medidas/Chains',modelo,'.mat'), 'resultados');
end