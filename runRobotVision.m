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
addpath 'wekaInterface';

%Se agrega dinamicamente el classpath the weka
dpath = {'wekaInterface/weka.jar'};
javaclasspath('-v1');
javaclasspath(dpath)

load('ConfigurationReleaseParticipants.mat')
%load('ConfigurationReleaseTiny.mat')

Configuration.visualizeImageInfo = false;
Configuration.useDepth = true;

showDatasetStats(Configuration);

[featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, objectsForTraining, objectsForTest] = featureExtraction (Configuration);

% Seleccion de caracteristicas
[featuresForTraining, featuresForTest] = seleccionCaracteristicas(Configuration, featuresForTraining, featuresForTest);
%aPriori(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest);

%Exportacion a arff
%extraerDataSet();
%convertirMisDatos();

%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 1:8, 'NB', true, true, true,0)
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 1:8, 'RL', true, true, true,0)
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 1:8, 'RF', true, true, true,0)

%busquedaHeuristica(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest,1,0);
%busquedaHeuristica(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest,0,8);

chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'RF');
