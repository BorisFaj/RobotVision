function runRobotVision
clear;

% ### INFORMATION ###
%
% PLACE THE TRAINING FILES AT
%
% RobotVision2014Data/trainingDepth
% and
% RobotVision2014Data/trainingVisual
%
% AND THE TEST FILES AT
%
% RobotVision2014Data/testDepth
% and
% RobotVision2014Data/testVisual
%
% All the configuration Information is included in the Configuratio struct
% that is loaded from ConfigurationReleaseParticipants.mat or ConfigurationReleaseTiny.mat
%
% Check all the parameteres that can be established in those files
%

close all;

addpath 'dogma';
addpath 'datos';
addpath 'wekaInterface';

%Se agrega dinamicamente el classpath the weka
dpath = {'wekaInterface/weka.jar'};
javaclasspath('-v1');
javaclasspath(dpath)

load('ConfigurationReleaseParticipants.mat')
%load('ConfigurationReleaseTiny.mat')
load('datos/featuresForTraining.mat')
load('datos/featuresForTest.mat')
load('datos/clasesForTraining.mat')
load('datos/clasesForTest.mat')
load('datos/objectsForTraining.mat')
load('datos/objectsForTest.mat')

Configuration.visualizeImageInfo = false;
Configuration.useDepth = true;

%Muestra informacion sobre el TESTset
showDatasetStats(Configuration);

%[featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, objectsForTraining, objectsForTest] = featureExtraction (Configuration);

%Exportacion a arff
%extraerDataSet();
%convertirMisDatos();

%Para hacer cross validation hay que juntar primero el dataset
features = [featuresForTraining;featuresForTest];
objects = [objectsForTraining objectsForTest];
rooms = [clasesForTraining;clasesForTest];

%Se divide y se empieza la clasificacion
obj = [cellstr('Obj1');cellstr('Obj2');cellstr('Obj3');cellstr('Obj4');cellstr('Obj5');cellstr('Obj6');cellstr('Obj7');cellstr('Obj8');cellstr('Total')];
et = [cellstr('-') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Error Rate') cellstr('TP%') cellstr('TN%') cellstr('FP%') cellstr('FN%')];
metricasOBNB = zeros(Configuration.numObjects+1,8);
metricasOBRL = zeros(Configuration.numObjects+1,8);
metricasOBC45 = zeros(Configuration.numObjects+1,8);
metricasOBSVM = zeros(Configuration.numObjects+1,8);
for ob=1:Configuration.numObjects
    CVO=cvpartition(objects(ob,:),'k',5);
    metricasCVNB = zeros(CVO.NumTestSets,8);
    metricasCVRL = zeros(Configuration.numObjects+1,8);
    metricasCVC45 = zeros(Configuration.numObjects+1,8);
    metricasCVSVM = zeros(Configuration.numObjects+1,8);
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);

        [train, test] = convertirAWeka(Configuration, features(trIdx,:), features(teIdx,:), objects(ob,trIdx), objects(ob,teIdx), 1);    
        metricasCVNB(i,:) = modelosWeka(Configuration, train, test, size(features,2)+1, 'NB', true, true);
        metricasCVRL(i,:) = modelosWeka(Configuration, train, test, size(features,2)+1, 'RL', true, true);
        metricasCVC45(i,:) = modelosWeka(Configuration, train, test, size(features,2)+1, 'C45', true, true);
        metricasCVSVM(i,:) = modelosWeka(Configuration, train, test, size(features,2)+1, 'SVM', true, true);
    end
    metricasOBNB(ob,:) = mean(metricasCVNB);
    metricasOBRL(ob,:) = mean(metricasCVRL);
    metricasOBC45(ob,:) = mean(metricasCVC45);
    metricasOBSVM(ob,:) = mean(metricasCVSVM);
end    
metricasOBNB(9,:) = mean(metricasOBNB(1:8,:));   
metricasOBNB = num2cell(metricasOBNB);
metricasOBNB = [obj metricasOBNB];
metricasOBNB = [et;metricasOBNB]
save('medidas/metricasNB', 'metricasOBNB');

metricasOBRL(9,:) = mean(metricasOBRL(1:8,:));   
metricasOBRL = num2cell(metricasOBRL);
metricasOBRL = [obj metricasOBRL];
metricasOBRL = [et;metricasOBRL]
save('medidas/metricasRL', 'metricasOBRL');

metricasOBC45(9,:) = mean(metricasOBC45(1:8,:));   
metricasOBC45 = num2cell(metricasOBC45);
metricasOBC45 = [obj metricasOBC45];
metricasOBC45 = [et;metricasOBC45]
save('medidas/metricasC45', 'metricasOBC45');

metricasOBSVM(9,:) = mean(metricasOBSVM(1:8,:));   
metricasOBSVM = num2cell(metricasOBSVM);
metricasOBSVM = [obj metricasOBSVM];
metricasOBSVM = [et;metricasOBSVM]
save('medidas/metricasSVM', 'metricasOBSVM');


% Seleccion de caracteristicas
%[featuresForTraining, featuresForTest] = seleccionCaracteristicas(Configuration, featuresForTraining, featuresForTest);

%mu = mean(featuresForTraining);
%sigma = std(featuresForTraining);
%X=bsxfun(@minus, featuresForTraining, mu);
%X=bsxfun(@rdivide, X, sigma);
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 1:8, 'RL', true, true, true,X);        
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 1:8, 'RF', true, true, true,0);        
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 1:8, 'NB', true, true, true,0);        

%busquedaHeuristica(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest,1,0);
%busquedaHeuristica(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest,0,8);

%chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'RF');

%clasificaHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, 'NB', true, false, 'e')
