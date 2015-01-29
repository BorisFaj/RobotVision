function [featuresForTraining, featuresForTest] = seleccionCaracteristicas(featuresForTraining, featuresForTest)
% Las caracteristicas elegidas utilizando weka con Cfs son:
%Objeto1: 3,31,33,57,66,69,85,125,239,257,258,260,261,262,263,264
%Objeto2: 1,4,8,14,60,70,213,233,255,257,258,259,260,263,264
%Objeto3: 2,3,4,7,14,16,35,72,117,160,186,192,204,242,254,255,257,258,259,260,262,263,264
%Objeto4: 2,74,199,255,257,258,262,263,264
%Objeto5: 1,9,11,37,48,52,65,69,70,74,118,219,252,258,259,260,262,264
%Objeto6: 3,4,7,13,160,258,262,263,264
%Objeto7: 2,3,4,5,7,14,17
%Objeto8: 3,263

  C = [1,2,3,4,5,7,8,9,11,13,14,16,17,31,33,35,37,48,52,57,60,65,66,69,70,72,74,85,117,118,125,160,186,192,199,204,213,219,233,239,242,252,254,255,257,258,259,260,261,262,263,264];
  
  for n=1:length(C)
      fTrAux(:,n) = featuresForTraining(:,C(n));
      fTestAux(:,n) = featuresForTest(:,C(n));
  end
  
  featuresForTraining = fTrAux;
  featuresForTest = fTestAux;
end