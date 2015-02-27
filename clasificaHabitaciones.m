function clasificaHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, modelo, depuracion, trampa, caracteristicas)
%
%   modelo: 'NB'|'RL'|'RF'|'DT'|'SVM'
%    (Naive Bayes, Regresion Logistica, Random Forest, Decision Tree, SVM)
%
%   caracteristicas: 'e'|'eyp'|'p'
%                (extraidas, extraidas + predichas, predichas)
%
%   Depuracion: {true|false}        carga las predicciones de un .mat
%   Trampa:     {true|false}        carga como predicciones las clases
%   reales


if(depuracion)
   if(trampa)
        load datos/objectsForTraining.mat
        load datos/objectsForTest.mat
        prediccionesTest = objectsForTest';
        prediccionesTraining = objectsForTraining';
   else
       load datos/prediccionesTestRF.mat;
       load datos/prediccionesTrainingRF.mat
       prediccionesTest=prediccionesTestRF;
       prediccionesTraining=prediccionesTrainingRF;
   end
end

if(strcmp('e',caracteristicas))
    %ya estan cargadas
elseif(strcmp('eyp',caracteristicas))
    featuresForTest = [featuresForTest prediccionesTest];
    featuresForTraining = [featuresForTraining prediccionesTraining];
elseif(strcmp('p',caracteristicas))
    featuresForTest = prediccionesTest;
    featuresForTraining = prediccionesTraining; 
else
    error('ERROR! El parametro caracteristicas de la funcion clasificaHabitaciones solo admite las cadenas: e|eyp|p (extraidas, extraidas + predichas, predichas)');
end

% Normalizar X para la RL
mu = mean(featuresForTraining);
sigma = std(featuresForTraining);
X=bsxfun(@minus, featuresForTraining, mu);
X=bsxfun(@rdivide, X, sigma);

for m=1:Configuration.numClasses
    if(strcmp('NB',modelo))
        %Entrenar modelo
        NB = fitNaiveBayes(featuresForTraining,categorical(clasesForTraining==m),'Distribution','kernel');            
        %Sacar las predicciones
        predicted = (double(NB.predict(featuresForTest))-1);
        %Puede generar valores NaN, los tomo como negativos
        predicted = predicted == 1;
    elseif(strcmp('RL',modelo))
        %Entrenar modelo
        RL = mnrfit(X,categorical(clasesForTraining==m));
        %Se predicen 2 columnas [falso positivo]
        predictedRL = mnrval(RL,featuresForTest);
        %Sacar las predicciones
        predicted = predictedRL(:,1) < 0.5;
    elseif(strcmp('RF',modelo))            
        %Entrenar modelo
        rng(1); % For reproducibility
        %Se calculan modelos con un numero de arboles de 1 a 50 con su
        %correspondiente out of the bag error
        RF = TreeBagger(50,featuresForTraining,categorical(double(clasesForTraining==m)),'OOBPred','On');   
        oobErr = oobError(RF);
        %Sacar las predicciones
        predicted = str2num(cell2mat(RF.predict(featuresForTest)));
	elseif(strcmp('DT',modelo))
        DT = fitctree(X,categorical(clasesForTraining==m));
        predicted = DT.predict(featuresForTest);
        predicted = predicted == categorical(1);
	elseif(strcmp('SVM',modelo))
        SVM = fitcsvm(X,categorical(clasesForTraining==m));
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
    accuracy = (TP+TN)/(TP+TN+FP+FN);
	%                 TP FP TN FN Recall Precision FScore clasificador
	resultados(m,:) = [TP FP TN FN recall precision FScore accuracy sum(actual)];
    end      	  
end

%Apaño los resultados
%Añado en la ultima fila las medias
resultados(11,1:4) = sum(resultados(:,1:4));
resultados(11,5) = resultados(11,1)/(resultados(11,1)+resultados(11,4)); %recall
resultados(11,6) = resultados(11,1)/(resultados(11,1)+resultados(11,2)); %precision
resultados(11,7) = 2*resultados(11,6)*resultados(11,5)/(resultados(11,6)+resultados(11,5)); %FScore
resultados(11,8) = (resultados(11,1)+resultados(11,3))/sum(resultados(11,1:4));
resultados(12,:) = resultados(11,:)
%Y en cada habitacion
for n=1:11
    %Los postivos
    total = resultados(n,1) + resultados(n,4);
    resultados(n,1) = resultados(n,1)/total;
    resultados(n,4) = resultados(n,4)/total;
    %Los negativos
    total = resultados(n,2) + resultados(n,3);
    resultados(n,2) = resultados(n,2)/total;
    resultados(n,3) = resultados(n,3)/total;
end
%Añado etiquetas
et = [cellstr('TP') cellstr('FP') cellstr('TN') cellstr('FN') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Accuracy') cellstr('num habitaciones')];
resultados = num2cell(resultados);
resultados = [et;resultados]

if(trampa)
    save(strcat('medidas/habitaciones',modelo,caracteristicas,'TRAMPA.mat'), 'resultados');
else
    save(strcat('medidas/habitaciones',modelo,caracteristicas,'.mat'), 'resultados');
end
end