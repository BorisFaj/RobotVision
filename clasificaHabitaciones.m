function [accuracy] = clasificaHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, modelo)
%
%   modelo: 'NB'|'RL'|'RF'|'DT'|'SVM'
%    (Naive Bayes, Regresion Logistica, Random Forest, Decision Tree, SVM)
%
%   caracteristicas: 'e'|'eyp'|'p'
%                (extraidas, extraidas + predichas, predichas)
%
%   Depuracion: {true|false}        modo depuracion: carga las predicciones de un .mat
%   Trampa:     {true|false}        carga como predicciones las clases
%   reales

% Normalizar X para la RL
mu = mean(featuresForTraining);
sigma = std(featuresForTraining);
X=bsxfun(@minus, featuresForTraining, mu);
X=bsxfun(@rdivide, X, sigma);

prediccionRL = zeros(length(clasesForTest),Configuration.numClasses);
prediccionC45 = zeros(length(clasesForTest),Configuration.numClasses);
if(strcmp('NB',modelo))
    %Entrenar modelo
    NB = fitNaiveBayes(featuresForTraining,categorical(clasesForTraining),'Distribution','kernel');            
    %Sacar las predicciones
    prediccionNB = NB.predict(featuresForTest);
    
    accuracy = numel(find(prediccionNB==categorical(clasesForTest)))/numel(clasesForTest)*100;
    
elseif(strcmp('DT',modelo))
    DT = fitctree(X,categorical(clasesForTraining),'SplitCriterion','deviance');
    prediccionC45 = DT.predict(featuresForTest);
    accuracy = numel(find(prediccionC45==categorical(clasesForTest)))/numel(clasesForTest)*100;
    
elseif(strcmp('RL',modelo))
    for m=1:Configuration.numClasses
        try
            %Entrenar modelo
            RL = mnrfit(X,categorical(clasesForTraining==m));
            %Se predicen 2 columnas [falso positivo]
            predictedRL = mnrval(RL,featuresForTest);
            %Sacar las predicciones, me quedo probabilidad de que sea FALSO
            prediccionRL(:,m) = predictedRL(:,1);
        catch ME
            warning('X and Y must contain at least one valid observation.');
        end       
    end
    [~,prediccion]=min(prediccionRL,[],2);
    accuracy = numel(find(prediccion==clasesForTest))/numel(clasesForTest)*100;    
end

end