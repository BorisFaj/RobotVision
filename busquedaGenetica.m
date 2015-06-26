load('datos/featuresForTraining.mat')
load('datos/featuresForTest.mat')
load('datos/objectsForTraining.mat')
load('datos/objectsForTest.mat')

%Población inicial
nIndividuos = 16;
poblacion = zeros(nIndividuos, 8);
x = 1:8;

for n=1:nIndividuos
    poblacion(n,:) = x(randperm(length(x)));
end

%Evaluación
fitness = zeros(1,nIndividuos);
for n=1:nIndividuos
    fitness(n) = chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'DT', poblacion(n,:));
end

%Selección por torneo
seleccionados = zeros(1,nIndividuos);
for n=1:nIndividuos
    [v, seleccionados(n)] = max(fitness(round(rand*nIndividuos)),fitness(round(rand*nIndividuos)));
end

%Cruce Two-pints center crossover (2PCX)
hijos = zeros(1,nIndividuos);
for n=1:nIndividuos/2
    p1 = round(rand*8);
    p2 = round(rand*8);    

    %El primer punto de cruce tiene que se menor
    if(p2>p1)
        aux = p1;
        p1 = p2;
        p2 = aux;
    end
    
    padre1= poblacion(seleccionados(n),:);
    padre2= poblacion(seleccionados(n)+(nIndividuos/2),:);
    
    %La primera parte de los padres se copia en los hijos
    hijo1 = padre1(1:p1);
    hijo2 = padre2(1:p1);
    
    %Apañar la parte del centro
    centro1=padre1(p1:p2);
    centro2=padre2(p1:p2);
    
    %Conseguir los indices del centro en el otro padre
    indices = zeros(2,length(centro1));
    for i=0:length(centro1)
        indices(1,i) = find(padre2==centro1(i));
        indices(2,i) = find(padre1==centro2(i));
    end
    
    %Ordenar los centros según el orden del otro padre
    for i=1:length(centro1)
        %Conseguir el indice mas bajo
        [~, aux1] = min(indices(1,:));
        [~, aux2] = min(indices(2,:));
        %Asegurar de que no va a volver a ser el mas bajo
        indices(1,aux1) = indices(1,aux1)+8;
        indices(2,aux2) = indices(2,aux2)+8;
        %Concatenar el valor e los hijos
        hijo1 = [hijo1, padre1(aux1)];
        hijo2 = [hijo2, padre2(aux2)];
    end
    
    %Concatenar la ultima parte de los padres
    hijo1 = [hijo1, padre1(p2:8)];
    hijo2 = [hijo2, padre1(p2:8)]; 
end

%Mutación

%Sustitución

%Parada