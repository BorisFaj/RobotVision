function regresionLogistica(Configuration, featuresForTraining, objectsForTraining)
    X = featuresForTraining;
    numVariables = size(X,2); % Lee el número de variables
    % Normalizar X
    %http://www.mathworks.com/matlabcentral/answers/21190-how-does-one-use-output-from-mnrfit-to-forecast-nominal-values
    % Normalizado según distribución normal!!!!
    %El error no se si es que la función de predicción está mal o que la distribucion es
    %multivariada (o las dos cosas)
    [X, mu, sigma] = normalizaCaracteristicas(X);
    
    objectsForTraining = categorical(objectsForTraining);
    
    modelos = zeros(Configuration.numObjects,numVariables+1);
    
    for m = 1:size(modelos,1)
        [modelos(m,:)] = mnrfit(X,objectsForTraining(m,:)','model','nominal')';
        m
    end
    
    save modelos.mat modelos;    
end

function [X_norm, mu, sigma] = normalizaCaracteristicas(X)
mu = mean(X);
sigma = std(X);

X_norm=bsxfun(@minus, X, mu);
X_norm=bsxfun(@rdivide, X_norm, sigma);
end