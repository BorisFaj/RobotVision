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

%Para hacer cross validation hay que juntar primero el dataset
features = [featuresForTraining;featuresForTest];
objects = [objectsForTraining objectsForTest];
rooms = [clasesForTraining;clasesForTest];

CVO=cvpartition(rooms,'k',5);
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    ytest = classify(features(teIdx,:),features(trIdx,:),rooms(trIdx,:),rooms(teIdx,:));
    err(i) = sum(~strcmp(ytest,species(teIdx)));
end
cvErr = sum(err)/sum(CVO.TestSize);

% Seleccion de caracteristicas
[featuresForTraining, featuresForTest] = seleccionCaracteristicas(Configuration, featuresForTraining, featuresForTest);


%Exportacion a arff
%extraerDataSet();
%convertirMisDatos();

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

clasificaHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, 'NB', true, false, 'e')
