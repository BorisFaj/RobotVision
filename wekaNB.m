function wekaNB(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest)

classindex = 53;
resultados = zeros(Configuration.numObjects+1,4);


for n = 1:(size(featuresForTraining,2))
    nombres(n) = cellstr(strcat('a',int2str(n)));
end
nombres(size(featuresForTraining,2)+1) = cellstr('clase');

for j=1:size(objectsForTraining,2)
    for i=1:size(objectsForTraining,1)
        if(objectsForTraining(i,j) == 1)
            clasesTrain(i,j)=cellstr('true');
        else
            clasesTrain(i,j)=cellstr('false');
        end
    end
end

for j=1:length(objectsForTest)
    for i=1:size(objectsForTest,1)
        if(objectsForTest(i,j) == 1)
            clasesTest(i,j)=cellstr('true');
        else
            clasesTest(i,j)=cellstr('false');
        end
    end
end

for m=1:Configuration.numObjects
    train = matlab2weka(strcat('obj',strcat(int2str(m),'-train')),nombres,[num2cell(featuresForTraining) clasesTrain(m,:)']);
    test =  matlab2weka(strcat('obj',strcat(int2str(m),'-test')),nombres,[num2cell(featuresForTest) clasesTest(m,:)']);

    %Train the classifier
    nb = trainWekaClassifier(train,'bayes.NaiveBayesMultinomial');

    %Test the classifier
    predicted = wekaClassify(test,nb);

    %The actual class labels (i.e. indices thereof)
    actual = test.attributeToDoubleArray(classindex-1); %java indexes from 0

    errorRate = sum(actual ~= predicted)/500

    TP=0;TN=0;FP=0;FN=0;

    for n=1:length(predicted)
        if(predicted(n)) %se predice positivo
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
    resultados(m,1) = TP/(TP+FN); %recall
    resultados(m,2) = TP/(TP+FP); %precision
    resultados(m,3) = 2*resultados(m,2)*resultados(m,1)/(resultados(m,2)+resultados(m,1)); %FScore
    resultados(m,4) = errorRate; %error rate
    
    resultados(Configuration.numObjects+1, 1) =resultados(Configuration.numObjects+1, 1)+resultados(m,1);
    resultados(Configuration.numObjects+1, 2) =resultados(Configuration.numObjects+1, 2)+resultados(m,2);
    resultados(Configuration.numObjects+1, 3) =resultados(Configuration.numObjects+1, 3)+resultados(m,3);
    resultados(Configuration.numObjects+1, 4) =resultados(Configuration.numObjects+1, 4)+resultados(m,4);
end

%Apañar la matriz de resultados
resultados(Configuration.numObjects+1, 1) =resultados(Configuration.numObjects+1, 1)/Configuration.numObjects;
resultados(Configuration.numObjects+1, 2) =resultados(Configuration.numObjects+1, 2)/Configuration.numObjects;
resultados(Configuration.numObjects+1, 3) =resultados(Configuration.numObjects+1, 3)/Configuration.numObjects;
resultados(Configuration.numObjects+1, 4) =resultados(Configuration.numObjects+1, 4)/Configuration.numObjects;

resultados = num2cell(resultados);
obj = [cellstr('Obj1');cellstr('Obj2');cellstr('Obj3');cellstr('Obj4');cellstr('Obj5');cellstr('Obj6');cellstr('Obj7');cellstr('Obj8');cellstr('Media')];
et = [cellstr('-') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Error Rate')];

resultados = [obj resultados];
resultados = [et;resultados]

save wekaNB.mat resultados
end