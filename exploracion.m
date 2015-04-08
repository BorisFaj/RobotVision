clear;

load('ConfigurationReleaseParticipants.mat')
%load('ConfigurationReleaseTiny.mat')

load('datos/featuresForTraining.mat')
load('datos/featuresForTest.mat')
load('datos/clasesForTraining.mat')
load('datos/clasesForTest.mat')
load('datos/objectsForTraining.mat')
load('datos/objectsForTest.mat')

addpath 'toolbox_ITL';

mayor = 256; % con profundidad 264, sin 256
medias = zeros(1,mayor);
desviacion = zeros(1,mayor);
rango = zeros(1,mayor);

for n=1:mayor
   medias(n) = mean(featuresForTraining(:,n)); 
   desviacion(n) = std(featuresForTraining(:,n)); 
   rango(n) = max(featuresForTraining(:,n)) - min(featuresForTraining(:,n));   
end

%Caracteristicas seleccionadas
%C = [1,2,3,4,5,7,8,9,11,13,14,16,17,31,33,35,37,48,52,57,60,65,66,69,70,72,74,85,117,118,125,160,186,192,199,204,213,219,233,239,242,252,254,255,257,258,259,260,261,262,263,264];
C = [1,2,3,4,5,7,8,9,11,13,14,16,17,31,33,35,37,48,52,57,60,65,66,69,70,72,74,85,117,118,125,160,186,192,199,204,213,219,233,239,242,252,254,255];

mediaC = medias(:,C);
desviacionC = desviacion(:,C);
rangoC = rango(:,C);

expC = [mediaC; desviacionC;rangoC];

%Caracteristicas NO seleccionadas
[temp] = 1:mayor;
NC = [];
for n=1:mayor
    if(sum(temp(n)==C)==0)
        NC = [NC, n];
    end
end
mediaNC = medias(:,NC);
desviacionNC = desviacion(:,NC);256
rangoNC = rango(:,NC);

expNC = [mediaNC; desviacionNC;rangoNC];

%Ganancia de informacion mutua

%Esta medida evalua segun la probabilidad de que se de un subconjunto dado
%(la clase pertenece a ese subconjunto). Al tratarse de variables continuas
%esta medida va a ser muy inexacta se haga como se haga. Aqui esta hecho a
%lo bestia pero tambien estaria la opcion de buscar el subconjunto optimo
%utilizando alguna metaheuristica o algoritmo voraz


%PROBAR CON NORMALIZACION


w = 0.05; % The sigma of Gauisssian
numEstimate = 100; % number of samples when using non-parametric pdf estimation

MI = zeros(8,mayor);
for n=1:8
    c = objectsForTraining(n,:); %clase para el objeto n
    MI(n,:) = calculateMIComplete(c,featuresForTraining(:,1:mayor),w,numEstimate);
    
    find(MI(n,:)>mean(MI(n,:))) %Indices de los valores por encima de la media
end