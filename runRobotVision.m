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


% %Para hacer cross validation hay que juntar primero el dataset
% features = [featuresForTraining;featuresForTest];
% objects = [objectsForTraining objectsForTest];
% rooms = [clasesForTraining;clasesForTest];
% 
% %Se divide y se empieza la clasificacion
% obj = [cellstr('Obj1');cellstr('Obj2');cellstr('Obj3');cellstr('Obj4');cellstr('Obj5');cellstr('Obj6');cellstr('Obj7');cellstr('Obj8');cellstr('Total')];
% et = [cellstr('-') cellstr('Recall') cellstr('Precision') cellstr('FScore') cellstr('Accuracy') cellstr('TP%') cellstr('TN%') cellstr('FP%') cellstr('FN%')];
% metricasOBNB = zeros(Configuration.numObjects+1,8);
% metricasOBRL = zeros(Configuration.numObjects+1,8);
% metricasOBC45 = zeros(Configuration.numObjects+1,8);
% metricasOBSVM = zeros(Configuration.numObjects+1,8);
% for ob=1:Configuration.numObjects
%     CVO=cvpartition(objects(ob,:),'k',5);
%     metricasCVNB = zeros(CVO.NumTestSets,8);
%     metricasCVRL = zeros(CVO.NumTestSets,8);
%     metricasCVC45 = zeros(CVO.NumTestSets,8);
%     metricasCVSVM = zeros(CVO.NumTestSets,8);
%     for i = 1:CVO.NumTestSets
%         trIdx = CVO.training(i);
%         teIdx = CVO.test(i);
% 
%         reducido = seleccionCaracteristicas(features, ob, 'C45');
%         [metricasCVC45(i,:), modeloC45] = modelosMatlab(Configuration, reducido(trIdx,:), reducido(teIdx,:), objects(ob,trIdx), objects(ob,teIdx), ob, 'DT', true, true);       
%         reducido = seleccionCaracteristicas(features, ob, 'NB');
%         [metricasCVNB(i,:), modeloNB] = modelosMatlab(Configuration, reducido(trIdx,:), reducido(teIdx,:), objects(ob,trIdx), objects(ob,teIdx), ob, 'NB', true, true); 
%         reducido = seleccionCaracteristicas(features, ob, 'RL');
%         [metricasCVRL(i,:), modeloRL] = modelosMatlab(Configuration, reducido(trIdx,:), reducido(teIdx,:), objects(ob,trIdx), objects(ob,teIdx), ob, 'RL', true, true);        
%         reducido = seleccionCaracteristicas(features, ob, 'SVM');
%         [metricasCVSVM(i,:), modeloSVM] = SVMObjetos(Configuration, reducido(trIdx,:), reducido(teIdx,:), objects(ob,trIdx), objects(ob,teIdx));                        
%     end
%     %Guarda los modelos en ficheros, un modelo por objeto
%     save(strcat('modelos/C45_',num2str(ob)), 'modeloC45');
%     save(strcat('modelos/NB_',num2str(ob)), 'modeloNB');
%     save(strcat('modelos/RL_',num2str(ob)), 'modeloRL');
%     save(strcat('modelos/SVM_',num2str(ob)), 'modeloSVM');
%     
%     %Calcular la media de las metricas de todos los folds
%     metricasOBNB(ob,:) = mean(metricasCVNB);
%     metricasOBRL(ob,:) = mean(metricasCVRL);
%     metricasOBC45(ob,:) = mean(metricasCVC45);
%     metricasOBSVM(ob,:) = mean(metricasCVSVM);
% end    
% metricasOBNB(9,:) = mean(metricasOBNB(1:8,:));   
% metricasOBNB = num2cell(metricasOBNB);
% metricasOBNB = [obj metricasOBNB];
% metricasOBNB = [et;metricasOBNB]
% save('medidas/metricasNB', 'metricasOBNB');
% 
% metricasOBRL(9,:) = mean(metricasOBRL(1:8,:));   
% metricasOBRL = num2cell(metricasOBRL);
% metricasOBRL = [obj metricasOBRL];
% metricasOBRL = [et;metricasOBRL]
% save('medidas/metricasRL', 'metricasOBRL');
% 
% metricasOBC45(9,:) = mean(metricasOBC45(1:8,:));   
% metricasOBC45 = num2cell(metricasOBC45);
% metricasOBC45 = [obj metricasOBC45];
% metricasOBC45 = [et;metricasOBC45]
% save('medidas/metricasC45', 'metricasOBC45');
% 
% metricasOBSVM(9,:) = mean(metricasOBSVM(1:8,:));   
% metricasOBSVM = num2cell(metricasOBSVM);
% metricasOBSVM = [obj metricasOBSVM];
% metricasOBSVM = [et;metricasOBSVM]
% save('medidas/metricasSVM', 'metricasOBSVM'); 
% 
% 
% 
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %                                                               %
% % %                   Clasificacion de habitaciones               %
% % %                                                               %
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %Se estratifica
% CVO=cvpartition(rooms,'k',5);
% %Inicializar los accuracy para todos los casos
% accuracySVMSoloC = zeros(1,CVO.NumTestSets);
% accuracyNBSoloC = zeros(1,CVO.NumTestSets);
% accuracyC45SoloC = zeros(1,CVO.NumTestSets);
% accuracyRLSoloC = zeros(1,CVO.NumTestSets);
% 
% accuracySVMSoloP = zeros(1,CVO.NumTestSets);
% accuracyNBSoloP = zeros(1,CVO.NumTestSets);
% accuracyC45SoloP = zeros(1,CVO.NumTestSets);
% accuracyRLSoloP = zeros(1,CVO.NumTestSets);
% 
% accuracySVMPC = zeros(1,CVO.NumTestSets);
% accuracyNBPC = zeros(1,CVO.NumTestSets);
% accuracyC45PC = zeros(1,CVO.NumTestSets);
% accuracyRLPC = zeros(1,CVO.NumTestSets);
% 
% accuracySVMSoloCl = zeros(1,CVO.NumTestSets);
% accuracyNBSoloCl = zeros(1,CVO.NumTestSets);
% accuracyC45SoloCl = zeros(1,CVO.NumTestSets);
% accuracyRLSoloCl = zeros(1,CVO.NumTestSets);
% 
% accuracySVMCC = zeros(1,CVO.NumTestSets);
% accuracyNBCC = zeros(1,CVO.NumTestSets);
% accuracyC45CC = zeros(1,CVO.NumTestSets);
% accuracyRLCC = zeros(1,CVO.NumTestSets);
% 
% for i = 1:CVO.NumTestSets
%     trIdx = CVO.training(i);
%     teIdx = CVO.test(i);
% 
%     %Solo Caracteristicas
%     accuracyRLSoloC(i) = clasificaHabitaciones(Configuration, features(trIdx,:), features(teIdx,:), rooms(trIdx), rooms(teIdx), 'RL');
%     accuracyNBSoloC(i) = clasificaHabitaciones(Configuration, features(trIdx,:), features(teIdx,:), rooms(trIdx), rooms(teIdx), 'NB');
%     accuracyC45SoloC(i) = clasificaHabitaciones(Configuration, features(trIdx,:), features(teIdx,:), rooms(trIdx), rooms(teIdx), 'DT');
%     accuracySVMSoloC(i) = SVMHabitaciones(Configuration, features(trIdx,:), features(teIdx,:), rooms(trIdx), rooms(teIdx));
%     
%     %Calcular predicciones
%     prediccionesNBTr = zeros(CVO.TrainSize(1),Configuration.numObjects);
%     prediccionesNBTe = zeros(CVO.TestSize(1),Configuration.numObjects);
%     prediccionesRLTr = zeros(CVO.TrainSize(1),Configuration.numObjects);
%     prediccionesRLTe = zeros(CVO.TestSize(1),Configuration.numObjects);
%     prediccionesC45Tr = zeros(CVO.TrainSize(1),Configuration.numObjects);
%     prediccionesC45Te = zeros(CVO.TestSize(1),Configuration.numObjects);
%     prediccionesSVMTr = zeros(CVO.TrainSize(1),Configuration.numObjects);
%     prediccionesSVMTe = zeros(CVO.TestSize(1),Configuration.numObjects);
%     for ob=1:Configuration.numObjects
%         load(strcat('modelos/C45_',num2str(ob)), 'modeloC45');
%         load(strcat('modelos/NB_',num2str(ob)), 'modeloNB');
%         load(strcat('modelos/RL_',num2str(ob)), 'modeloRL');
%         load(strcat('modelos/SVM_',num2str(ob)), 'modeloSVM'); 
%         
%         %Predecir NB
%         predicted = (double(modeloNB.predict(features(trIdx,:)))-1);
%         prediccionesNBTr(:,ob) = predicted == 1;
%         
%         predicted = (double(modeloNB.predict(features(teIdx,:)))-1);
%         prediccionesNBTe(:,ob) = predicted == 1;
%         
%         %Predecir RL
%         predicted = mnrval(modeloRL,features(trIdx,:));
%         predicted = predicted(:,1);
%         prediccionesRLTr(:,ob) = predicted<=0.5;
%         
%         predicted = mnrval(modeloRL,features(teIdx,:));
%         predicted = predicted(:,1);
%         prediccionesRLTe(:,ob) = predicted<=0.5;
%         
%         %Predecir C45
%         predicted = modeloC45.predict(features(trIdx,:));    
%         prediccionesC45Tr(:,ob) = predicted == categorical(1);     
%         
%         predicted = modeloC45.predict(features(teIdx,:));    
%         prediccionesC45Te(:,ob) = predicted == categorical(1);   
%         
%         %Predecir SVM
%         [pred_oisvm_last,~]=model_predict(features(trIdx,:)',modeloSVM,0);
%         auxpred=pred_oisvm_last;
%         for n=1:numel(pred_oisvm_last)
%             if(pred_oisvm_last(n)==-1)
%                 auxpred(n)=0;
%             else
%                 auxpred(n)=1;
%             end
%         end
%         prediccionesSVMTr(:,ob) = auxpred;
%         
%         [pred_oisvm_last,~]=model_predict(features(teIdx,:)',modeloSVM,0);
%         auxpred=pred_oisvm_last;
%         for n=1:numel(pred_oisvm_last)
%             if(pred_oisvm_last(n)==-1)
%                 auxpred(n)=0;
%             else
%                 auxpred(n)=1;
%             end
%         end
%         prediccionesSVMTe(:,ob) = auxpred;
% 
%     end
%     
%     %Solo Predicciones
%     accuracyRLSoloP(i) = clasificaHabitaciones(Configuration, prediccionesRLTr, prediccionesRLTe, rooms(trIdx), rooms(teIdx), 'RL');
%     accuracyNBSoloP(i) = clasificaHabitaciones(Configuration, prediccionesNBTr, prediccionesNBTe, rooms(trIdx), rooms(teIdx), 'NB');
%     accuracyC45SoloP(i) = clasificaHabitaciones(Configuration, prediccionesC45Tr, prediccionesC45Te, rooms(trIdx), rooms(teIdx), 'DT');
%     accuracySVMSoloP(i) = SVMHabitaciones(Configuration, prediccionesSVMTr, prediccionesSVMTe, rooms(trIdx), rooms(teIdx));
%     
%     %Caracteristicas + Predicciones
%     accuracyRLPC(i) = clasificaHabitaciones(Configuration, [features(trIdx,:), prediccionesRLTr], [features(teIdx,:), prediccionesRLTe], rooms(trIdx), rooms(teIdx), 'RL');
%     accuracyNBPC(i) = clasificaHabitaciones(Configuration, [features(trIdx,:), prediccionesNBTr], [features(teIdx,:), prediccionesNBTe], rooms(trIdx), rooms(teIdx), 'NB');
%     accuracyC45PC(i) = clasificaHabitaciones(Configuration, [features(trIdx,:), prediccionesC45Tr], [features(teIdx,:), prediccionesC45Te], rooms(trIdx), rooms(teIdx), 'DT');
%     accuracySVMPC(i) = SVMHabitaciones(Configuration, [features(trIdx,:), prediccionesSVMTr], [features(teIdx,:), prediccionesSVMTe], rooms(trIdx), rooms(teIdx));
%     
%     %Solo Clases
%     accuracyRLSoloCl(i) = clasificaHabitaciones(Configuration, objects(:,trIdx)', objects(:,teIdx)', rooms(trIdx), rooms(teIdx), 'RL');
%     accuracyNBSoloCl(i) = clasificaHabitaciones(Configuration, objects(:,trIdx)', objects(:,teIdx)', rooms(trIdx), rooms(teIdx), 'NB');
%     accuracyC45SoloCl(i) = clasificaHabitaciones(Configuration, objects(:,trIdx)', objects(:,teIdx)', rooms(trIdx), rooms(teIdx), 'DT');
%     accuracySVMSoloCl(i) = SVMHabitaciones(Configuration, objects(:,trIdx)', objects(:,teIdx)', rooms(trIdx), rooms(teIdx));
%     
%     %Caracteristicas + Clases
%     accuracyRLCC(i) = clasificaHabitaciones(Configuration, [features(trIdx,:), objects(:,trIdx)'], [features(teIdx,:), objects(:,teIdx)'], rooms(trIdx), rooms(teIdx), 'RL');
%     accuracyNBCC(i) = clasificaHabitaciones(Configuration, [features(trIdx,:), objects(:,trIdx)'], [features(teIdx,:), objects(:,teIdx)'], rooms(trIdx), rooms(teIdx), 'NB');
%     accuracyC45CC(i) = clasificaHabitaciones(Configuration, [features(trIdx,:), objects(:,trIdx)'], [features(teIdx,:), objects(:,teIdx)'], rooms(trIdx), rooms(teIdx), 'DT');
%     accuracySVMCC(i) = SVMHabitaciones(Configuration, [features(trIdx,:), objects(:,trIdx)'], [features(teIdx,:), objects(:,teIdx)'], rooms(trIdx), rooms(teIdx));
% end
% 
% save('medidas/accuracySVMSoloC', 'accuracySVMSoloC'); 
% save('medidas/accuracyNBSoloC', 'accuracyNBSoloC'); 
% save('medidas/accuracyC45SoloC', 'accuracyC45SoloC'); 
% save('medidas/accuracyRLSoloC', 'accuracyRLSoloC'); 
% 
% save('medidas/accuracySVMSoloP', 'accuracySVMSoloP'); 
% save('medidas/accuracyNBSoloP', 'accuracyNBSoloP'); 
% save('medidas/accuracyC45SoloP', 'accuracyC45SoloP'); 
% save('medidas/accuracyRLSoloP', 'accuracyRLSoloP'); 
% 
% save('medidas/accuracySVMPC', 'accuracySVMPC'); 
% save('medidas/accuracyNBPC', 'accuracyNBPC'); 
% save('medidas/accuracyC45PC', 'accuracyC45PC'); 
% save('medidas/accuracyRLPC', 'accuracyRLPC'); 
% 
% save('medidas/accuracySVMSoloCl', 'accuracySVMSoloCl'); 
% save('medidas/accuracyNBSoloCl', 'accuracyNBSoloCl'); 
% save('medidas/accuracyC45SoloCl', 'accuracyC45SoloCl'); 
% save('medidas/accuracyRLSoloCl', 'accuracyRLSoloCl'); 
% 
% save('medidas/accuracySVMCC', 'accuracySVMCC'); 
% save('medidas/accuracyNBCC', 'accuracyNBCC'); 
% save('medidas/accuracyC45CC', 'accuracyC45CC'); 
% save('medidas/accuracyRLCC', 'accuracyRLCC'); 

% Ranking de objetos para clasificar habitaciones con CFS 1,2,5,6,3,4,8,7
ordenCFS = [1,2,5,6,3,4,8,7];

chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'SVM', ordenCFS);
% chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'NB', ordenCFS);
% chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'DT', ordenCFS);
% chainsClassifier(Configuration, featuresForTraining, featuresForTest, objectsForTraining, objectsForTest, 'RL', ordenCFS);

%clasificaHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, 'NB', true, false, 'e')
