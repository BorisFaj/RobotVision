function clasificarRegresion(Configuration, featuresForTest, objectsForTest)
    load('modelos.mat');
    resultados = zeros(Configuration.numObjects+1,8);
    % Clasificacion
    X = featuresForTest;
    y = objectsForTest;
    % Normalizar X
    [X, mu, sigma] = normalizaCaracteristicas(X);
    % Añadir X_0
    X = [ones(length(y),1) X];
    % Umbral
    umbral=0.5;
    % Evaluar resultados    
    for n = 1:length(y)
        for m = 1:Configuration.numObjects        
            h = sigmoide(X(n,:)*modelos(m,:)');
            if(h>=umbral)   % se clasifica positivo
                if(y(m,n) == 1) % es positivo
                    resultados(m,1)=resultados(m,1)+1; %TP
                else
                    resultados(m,2)=resultados(m,2)+1; %FP
                end
            else           % se clasifica negativo
                if(y(m,n) == 0) % es negativo
                    resultados(m,3)=resultados(m,3)+1; %TN
                else
                    resultados(m,4)=resultados(m,4)+1; %FN
                end  
            end
        end
    end
    
    % Metricas [TP, TN, FP, FN, precision, recall, F1Score]
    for n=1:Configuration.numObjects
        % Se calculan las que faltan
        resultados(n,5) = resultados(n,1)/(resultados(n,1)+resultados(n,4)); %recall = TP/(TP+FN)
        resultados(n,6) = resultados(n,1)/(resultados(n,1)+resultados(n,2)); %precision = TP/(TP+FP)
        resultados(n,7) = 2*resultados(n,6)*resultados(n,5)/(resultados(n,5)+resultados(n,6)); %FScore = 2*precision*recall/(precision+recall)
        resultados(n,8) = (resultados(n,1)+resultados(n,3))/(resultados(n,1)+resultados(n,4)+resultados(n,2)+resultados(n,3)); %Tasa de acierto = TP+TN/P+N
        % Se normalizan las metricas que ya hay
        totalP = resultados(n,1)+resultados(n,4); %TP+FN
        totalN =  resultados(n,3)+resultados(n,2); %TN+FP
        resultados(n,1) = resultados(n,1)/totalP; %TP
        resultados(n,2) = resultados(n,2)/totalN; %FP
        resultados(n,3) = resultados(n,3)/totalN; %TN
        resultados(n,4) = resultados(n,4)/totalP; %FN
    end   
    
    % Media total
    for n=1:8
        resultados(Configuration.numObjects+1,n) = sum(resultados(:,n))/Configuration.numObjects;
    end
    resultados = num2cell(resultados);
    etiquetasX = [cellstr('TP%'),cellstr('FP%'),cellstr('TN%'),cellstr('FN%'),cellstr('recall'),cellstr('precision'),cellstr('FScore'),cellstr('Tasa de acierto')];
    etiquetasY = [cellstr('-');cellstr('Ojb1');cellstr('Ojb2');cellstr('Ojb3');cellstr('Ojb4');cellstr('Ojb5');cellstr('Ojb6');cellstr('Ojb7');cellstr('Ojb8');cellstr('Media')];
    resultados = [etiquetasX;resultados];
    resultados = [etiquetasY,resultados];
    resultados = [resultados(:,1:2),resultados(:,4),resultados(:,7),resultados(:,6),resultados(:,8:9)]
    save resultadosRegresion.mat resultados    
end

function [X_norm, mu, sigma] = normalizaCaracteristicas(X)
mu = mean(X);
sigma = std(X);

X_norm=bsxfun(@minus, X, mu);
X_norm=bsxfun(@rdivide, X_norm, sigma);
end

function g = sigmoide(z)
g = 1./(1+(exp(1).^-z));
end