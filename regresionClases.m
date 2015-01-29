function [results] = regresionClases(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest,lambda, num_iters, alpha, results)

X = featuresForTraining;
y = clasesForTraining;
numVariables = size(X,2); % Lee el número de variables

% Normalizar X
[X, mu, sigma] = normalizaCaracteristicas(X);

% Añadir X_0
m = length(y); % Número de ejemplos de entrada
X = [ones(m,1) X];

% Se construyen m modelos
modelos = zeros(Configuration.numClasses,numVariables+1);

for m = 1:size(modelos,1)
%for m = 1:0
    yAux = zeros(length(y),1);
    for i=1:length(y)
        if(y(i) == m)
            yAux(i) = 1;
        else
            yAux(i) = 0;
        end
    end
    % Gradiente descendiente
    [modelos(m,:)] = gradienteDescendiente(X, yAux, rand(numVariables+1, 1), alpha, lambda, num_iters);
    m
end

save modelos.mat modelos
%modelos = load('modelos.mat');
%modelos = modelos.modelos;

% Clasificacion
X = featuresForTest;
y = clasesForTest;
% Normalizar X
[X, mu, sigma] = normalizaCaracteristicas(X);
% Añadir X_0
X = [ones(length(y),1) X];
for n = 1:length(y)
    h=0;
    for m = 1:size(modelos,1)
        hn = sigmoide(X(n,:)*modelos(m,:)');
        if(hn>h)
            h = hn;
            results(n,1) = m;
        end
    end
end
end

%Tambien hay que mirar si las iteraciones son suficientes para que converja
function score = evalua(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest,lambda, num_iters, alpha, results)
%score = zeros(100,3);    
lambda = 0;
load('score.mat');
for l=1:100
    X = featuresForTraining;
    y = clasesForTraining;
    numVariables = size(X,2); % Lee el número de variables

    % Normalizar X
    [X, mu, sigma] = normalizaCaracteristicas(X);

    % Añadir X_0
    m = length(y); % Número de ejemplos de entrada
    X = [ones(m,1) X];

    % Se construyen m modelos
    modelos = zeros(Configuration.numClasses,numVariables+1);

    for m = 1:size(modelos,1)
    %for m = 1:0
        yAux = zeros(length(y),1);
        for i=1:length(y)
            if(y(i) == m)
                yAux(i) = 1;
            else
                yAux(i) = 0;
            end
        end
        % Gradiente descendiente
        [modelos(m,:)] = gradienteDescendiente(X, yAux, rand(numVariables+1, 1), alpha, lambda, num_iters);
        m
    end

    save modelos.mat modelos
    %modelos = load('modelos.mat');
    %modelos = modelos.modelos;

    % Clasificacion
    X = featuresForTest;
    y = clasesForTest;
    % Normalizar X
    [X, mu, sigma] = normalizaCaracteristicas(X);
    % Añadir X_0
    X = [ones(length(y),1) X];
    bien = 0;
    mal = 0;

    for n = 1:length(y)
        h=0;
        for m = 1:size(modelos,1)
            hn = sigmoide(X(n,:)*modelos(m,:)');
            if(hn>h)
                h = hn;
                results(n,1) = m;
            end
        end
        if(y(n)==results(n,1))
            bien = bien+1;
        else
            mal = mal+1;
        end
    end                
    [score(l,:)] = [mal, bien, lambda]       
    save score.mat score
    lambda = lambda+1;
    mal = 0;
    bien =0;
    l
end

end

function [theta, J_traza] = gradienteDescendiente(X, y, theta, alpha, lambda, num_iters)
jAnt = 0;
jN = 0;
iter=0;
while iter < num_iters,
    thetaAux = theta;
    thetaAux(1) = theta(1) - ((alpha/length(y)) * sum(X(:,1)'*(sigmoide(X*theta)-y)));
    for var=2:size(X,2)
        thetaAux(var) = theta(var)*(1-(alpha*lambda)/length(y)) - (alpha/length(y)) * sum(X(:,var)'*(sigmoide(X*theta)-y));
    end
    theta = thetaAux;
    jN = coste(X, y, theta, lambda);
    if((abs(jAnt-jN) > 0.00001) || (jN == Inf) || (jAnt == Inf) || (isnan(jN)) || (isnan(jAnt)))
        jAnt = jN;
    else
        iter
        iter=num_iters;
    end
    iter = iter+1;
end
end

function g = sigmoide(z)
g = 1./(1+(exp(1).^-z));
end

function [X_norm, mu, sigma] = normalizaCaracteristicas(X)
mu = mean(X);
sigma = std(X);

X_norm=bsxfun(@minus, X, mu);
X_norm=bsxfun(@rdivide, X_norm, sigma);
end

function J = coste(X, y, theta, lambda)
thetaPen = theta([2:length(theta)],1);
J = (1/length(y))*(-y'*log(sigmoide(X*theta))-(1-y)'*log(1-sigmoide(X*theta))) + lambda/(2*length(y)).*(thetaPen' * thetaPen);
end
