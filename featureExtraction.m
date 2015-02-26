function [featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, objectsForTraining, objectsForTest] = featureExtraction(Configuration)
%
% Falta extraer la informacion del validation set. Para eso, primero hay
% que adaptar el fichero de configuracion y despues en este script hacer lo
% mismo que para los otros datasets

showSurface = true;
% Visualization screens

if(Configuration.visualizeImageInfo && Configuration.useDepth)
    if(showSurface)
        ref1 = figure('Name','Top View - You can rotate','NumberTitle','off','Color', [1 1 1]);
    else
        ref1 = -1;
    end
    ref2 = figure('Name','Image Information','NumberTitle','off','Color', [1 1 1]);
end

% Histogram for depth

binSize = (8/Configuration.binsDepth);
binFirstCentroid = binSize/2;

% Estimate the number of training and test frames from sampling values

trainingFiles = floor(Configuration.trainingSamplingRate * Configuration.trainingFiles);
validationFiles = floor(Configuration.validationSamplingRate * Configuration.validationFiles);
testFiles = floor(Configuration.testSamplingRate * Configuration.testFiles);


% Pre allocate the default features structs

if strcmp(Configuration.visualFeatures,'histo')
    
    visualFeaturesTraining = zeros(trainingFiles,256);
    visualFeaturesValidation = zeros(testFiles,256);
    visualFeaturesTest = zeros(testFiles,256);
end

if strcmp(Configuration.depthFeatures,'histo')
    
    depthFeaturesTraining = zeros(trainingFiles,Configuration.binsDepth);
    depthFeaturesValidation = zeros(validationFiles,Configuration.binsDepth);
    depthFeaturesTest = zeros(testFiles,Configuration.binsDepth);
end

% Training Visual

% Features are loaded when previously generated

if (exist('RobotVision2014Data/trainingFeatures/','dir') ~= 7)
    mkdir ('RobotVision2014Data/', 'trainingFeatures');
end

featureType = strcat('RobotVision2014Data/trainingFeatures/',Configuration.visualFeatures, 'Visual/');
if (exist(featureType,'dir')~= 7)
    mkdir (featureType);
end;

for i=1:trainingFiles
    if strcmp(Configuration.visualFeatures,'histo')
        
        featureType = strcat('RobotVision2014Data/trainingFeatures/',Configuration.visualFeatures, 'Visual/');
        featureName = strcat(featureType, Configuration.trainingInfo.visualPath{i});
        featureName = strcat(featureName(1:end-3), 'mat');
        
        if exist(featureName, 'file') == 2
            load(featureName);
            visualFeaturesTraining(i,:) = histo;
        else
            infoImgs = imread(strcat('RobotVision2014Data/trainingVisual/',Configuration.trainingInfo.visualPath{i}));
            grayImg=mat2gray(rgb2gray(infoImgs));
            a = imhist(grayImg);
            a = a/sum(a);
            visualFeaturesTraining(i,:) = a';
            histo=a';
            save(featureName, 'histo');
        end
    end
end

% Training Depth

% Features are loaded when previously generated

if(exist('RobotVision2014Data/trainingFeatures/','dir')~= 7)
    mkdir ('RobotVision2014Data/', 'trainingFeatures');
end

featureType = strcat('RobotVision2014Data/trainingFeatures/',Configuration.depthFeatures, 'Depth/');
if (exist(featureType,'dir') ~= 7)
    mkdir (featureType);
end

if(Configuration.useDepth)
    for i=1:trainingFiles
        if strcmp(Configuration.depthFeatures,'histo')
            featureType = strcat('RobotVision2014Data/trainingFeatures/',Configuration.depthFeatures, 'Depth/');
            featureName = strcat(featureType, Configuration.trainingInfo.depthPath{i});
            featureName = strcat(featureName(1:end-3), 'mat');
            
            if(Configuration.visualizeImageInfo)
                points3d = loadpcd(strcat('RobotVision2014Data/trainingDepth/',Configuration.trainingInfo.depthPath{i}));
                visualize3DInfo(points3d(:,:,3),points3d(:,:,5),points3d(:,:,6),points3d(:,:,7),ref1,ref2,showSurface);
            end
            
            if exist(featureName, 'file') == 2
                load(featureName);
                depthFeaturesTraining(i,:) = histo;
            else
                
                if(Configuration.visualizeImageInfo==false)
                    points3d = loadpcd(strcat('RobotVision2014Data/trainingDepth/',Configuration.trainingInfo.depthPath{i}));
                end
                
                % DEFAULT IMPLEMENTATION --> DEPTH HISTOGRAM
                
                aux=points3d(:,:,3);
                a = hist(aux(:),binFirstCentroid:binSize:8);
                a = a/sum(a);
                depthFeaturesTraining(i,:) = a;
                histo=a';
                save(featureName, 'histo');
            end
        end
    end
end

% Test Visual

% Features are loaded when previously generated

if (exist('RobotVision2014Data/testFeatures/','dir') ~= 7)
    mkdir ('RobotVision2014Data/', 'testFeatures');
end

featureType = strcat('RobotVision2014Data/testFeatures/',Configuration.visualFeatures, 'Visual/');
if (exist(featureType,'dir') ~= 7)
    mkdir (featureType);
end;

for i=1:testFiles
    if strcmp(Configuration.visualFeatures,'histo')
        featureType = strcat('RobotVision2014Data/testFeatures/',Configuration.visualFeatures, 'Visual/');
        featureName = strcat(featureType, Configuration.testInfo.visualPath{i});
        featureName = strcat(featureName(1:end-3), 'mat');
        
        if exist(featureName, 'file') == 2
            load(featureName);
            visualFeaturesTest(i,:) = histo;
        else
            infoImgs = imread(strcat('RobotVision2014Data/testVisual/',Configuration.testInfo.visualPath{i}));
            grayImg=mat2gray(rgb2gray(infoImgs));
            a = imhist(grayImg);
            a = a/sum(a);
            visualFeaturesTest(i,:) = a';
            histo=a';
            save(featureName, 'histo');
        end
    end
end

% Test Depth

% Features are loaded when previously generated

if (exist('RobotVision2014Data/testFeatures/','dir') ~= 7)
    mkdir ('RobotVision2014Data/', 'testFeatures');
end

featureType = strcat('RobotVision2014Data/testFeatures/',Configuration.depthFeatures, 'Depth/');
if (exist(featureType,'dir') ~= 7)
    mkdir (featureType);
end

if(Configuration.useDepth)
    for i=1:testFiles
        if strcmp(Configuration.depthFeatures,'histo')
            featureType = strcat('RobotVision2014Data/testFeatures/',Configuration.depthFeatures, 'Depth/');
            featureName = strcat(featureType, Configuration.testInfo.depthPath{i});
            featureName = strcat(featureName(1:end-3), 'mat');
            
            if(Configuration.visualizeImageInfo)
                points3d = loadpcd(strcat('RobotVision2014Data/testDepth/',Configuration.testInfo.depthPath{i}));
                visualize3DInfo(points3d(:,:,3),points3d(:,:,5),points3d(:,:,6),points3d(:,:,7),ref1,ref2,showSurface);
            end
            
            if exist(featureName, 'file') == 2
                load(featureName);
                depthFeaturesTest(i,:) = histo;
            else
                
                if(Configuration.visualizeImageInfo==false)
                    points3d = loadpcd(strcat('RobotVision2014Data/testDepth/',Configuration.testInfo.depthPath{i}));
                end
                
                aux=points3d(:,:,3);
                a = hist(aux(:),binFirstCentroid:binSize:8);
                a = a/sum(a);
                depthFeaturesTest(i,:) = a;
                histo=a';
                save(featureName, 'histo');
            end
        end
    end
end

% Concatenate visual and depth labels in a single descriptor

if(Configuration.useDepth)
    featuresForTraining = [visualFeaturesTraining depthFeaturesTraining];
    featuresForTest  =  [visualFeaturesTest depthFeaturesTest];
else
    featuresForTest  = visualFeaturesTest;
    featuresForTraining = visualFeaturesTraining;
end

% Extract the labels for the room cathegory and for the objects appearance

clasesForTraining = Configuration.trainingInfo.class(1:trainingFiles,:);
clasesForTest = Configuration.testInfo.class(1:testFiles,:);

objectsForTraining = Configuration.trainingInfo.objectAppearance(:,1:trainingFiles);
objectsForTest = Configuration.testInfo.objectAppearance(:,1:testFiles);