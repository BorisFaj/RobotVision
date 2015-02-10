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

%Exportacion a arff
%extraerDataSet();
%convertirMisDatos();

%El ultimo parametro va true si hay que re-calcular. Si va false lo lee de
%los ficheros.
%[train, test] = convertirAWeka(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, false);
%modelosWeka(Configuration, train, test, size(featuresForTraining,2)+1, 'RF', true, true);
%modelosWeka(Configuration, train, test, size(featuresForTraining,2)+1, 'RL', true, true);
%modelosWeka(Configuration, train, test, size(featuresForTraining,2)+1, 'NB', true, true);
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'NB', true, true)
modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'RL', true, true)
%modelosMatlab(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'RF', true, true)
busquedaHeuristica(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest);
