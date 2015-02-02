function [train, test] = convertirAWeka(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, calcular)

if(calcular)
    % Lo primero que hay que hacer es convertir la clase a nominal
    nombres = cell(size(featuresForTraining,2)+1,1);

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

    % Y entonces convertir los datos al formato
    for m=1:Configuration.numObjects
        train(m) = matlab2weka(strcat('obj',strcat(int2str(m),'-train')),nombres,[num2cell(featuresForTraining) clasesTrain(m,:)']);
        test(m) =  matlab2weka(strcat('obj',strcat(int2str(m),'-test')),nombres,[num2cell(featuresForTest) clasesTest(m,:)']);
        m
    end
    save trainWeka.mat train;
    save testWeka.mat test
else
    load('trainWeka.mat');
    load('testWeka.mat');
end
end