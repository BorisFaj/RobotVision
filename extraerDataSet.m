for j=1:size(objectsForTraining,2)
    for i=1:size(objectsForTraining,1)
        if(objectsForTraining(i,j) == 1)
            clasesTrain(i,j)=cellstr('true');
        else
            clasesTrain(i,j)=cellstr('false');
        end
    end
end

objectsForTraining = clasesTrain;
featuresForTraining = num2cell(featuresForTraining);

Obj1 = [featuresForTraining,objectsForTraining(1,:)'];
Obj2 = [featuresForTraining,objectsForTraining(2,:)'];
Obj3 = [featuresForTraining,objectsForTraining(3,:)'];
Obj4 = [featuresForTraining,objectsForTraining(4,:)'];
Obj5 = [featuresForTraining,objectsForTraining(5,:)'];
Obj6 = [featuresForTraining,objectsForTraining(6,:)'];
Obj7 = [featuresForTraining,objectsForTraining(7,:)'];
Obj8 = [featuresForTraining,objectsForTraining(8,:)'];

save 'wekaInterface/Obj1.mat' Obj1
save 'wekaInterface/Obj2.mat' Obj2
save 'wekaInterface/Obj3.mat' Obj3
save 'wekaInterface/Obj4.mat' Obj4
save 'wekaInterface/Obj5.mat' Obj5
save 'wekaInterface/Obj6.mat' Obj6
save 'wekaInterface/Obj7.mat' Obj7
save 'wekaInterface/Obj8.mat' Obj8

for j=1:length(objectsForTest)
    for i=1:size(objectsForTest,1)
        if(objectsForTest(i,j) == 1)
            clasesTest(i,j)=cellstr('true');
        else
            clasesTest(i,j)=cellstr('false');
        end
    end
end

objectsForTest = clasesTest;
featuresForTest = num2cell(featuresForTest);

Test1 = [featuresForTest,objectsForTest(1,:)'];
Test2 = [featuresForTest,objectsForTest(2,:)'];
Test3 = [featuresForTest,objectsForTest(3,:)'];
Test4 = [featuresForTest,objectsForTest(4,:)'];
Test5 = [featuresForTest,objectsForTest(5,:)'];
Test6 = [featuresForTest,objectsForTest(6,:)'];
Test7 = [featuresForTest,objectsForTest(7,:)'];
Test8 = [featuresForTest,objectsForTest(8,:)'];

save 'wekaInterface/Test1.mat' Test1
save 'wekaInterface/Test2.mat' Test2
save 'wekaInterface/Test3.mat' Test3
save 'wekaInterface/Test4.mat' Test4
save 'wekaInterface/Test5.mat' Test5
save 'wekaInterface/Test6.mat' Test6
save 'wekaInterface/Test7.mat' Test7
save 'wekaInterface/Test8.mat' Test8