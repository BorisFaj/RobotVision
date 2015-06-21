function [resultados model_oisvm] = SVMObjetos(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest)

verbose = false;

% The object recognition is made using a binary classificator

% DOGMA CONFIGURAION

hp.type='expchi2';
hp.gamma= 1;
hp.coef0 = 10;
hp.degree = 2;

%       'linear'              Linear kernel or dot product
%       'poly'                Polynomial kernel
%       'rbf'                 Gaussian Radial Basis Function kernel
%       'sigmoid'             Sigmoidal kernel
%       'triangular'          Triangular kernel
%       'intersection'        Intersection kernel
%       'intersection_sparse' Intersection kernel for sparse matrices
%       'expchi2'             Exponential Chi^2 kernel
%       'expchi2_sparse'      Exponential Chi^2 kernel for sparse matrices


%inizialize an empty model
model_bak=model_init(@compute_kernel,hp);

% choose the parameter 'eta' of the OISVM
model_bak.eta=0.25;
% choose the parameter 'C' of the OISVM
model_bak.C=1;

model_bak.step = 250;

x_tr = featuresForTraining;
x_te = featuresForTest;

    
% We generate the training and testing sequences using +1 or -1, taking
% into account just the current class

y = clasesForTraining;

for i=1:numel(y)
    if(y(i)==0)
        y(i)=-1;
    else
        y(i)=1;
    end
end    

y_tr = y;

%fflush(stdout);

model_oisvm=k_oisvm_train(x_tr',y_tr',model_bak);    
[pred_oisvm_last,~]=model_predict(x_te',model_oisvm,0);

auxpred=pred_oisvm_last;

for i=1:numel(pred_oisvm_last)
    if(pred_oisvm_last(i)==-1)
        auxpred(i)=0;
    else
        auxpred(i)=1;
    end
end

%Sacar las metricas
TP=0;TN=0;FP=0;FN=0;
for n=1:length(auxpred)
    if(auxpred(n)) %se predice positivo
        if(clasesForTest(n))    %es positivo
            TP=TP+1;
        else
            FP = FP+1;
        end
    else            %se predice negativo
        if(~clasesForTest(n))  %es negativo
            TN=TN+1;
        else
            FN=FN+1;
        end
    end
end
resultados(1) = TP/(TP+FN); %recall
resultados(2) = TP/(TP+FP); %precision
resultados(3) = 2*resultados(2)*resultados(1)/(resultados(2)+resultados(1)); %FScore
resultados(4) = (TP+TN)/(TP+TN+FP+FN); %accuracy
resultados(5) = TP; %TP
resultados(6) = TN; %TN
resultados(7) = FP; %FP
resultados(8) = FN; %FN

%Sacar las tasas
totalP = TP+FN;
totalN = TN+FP;
resultados(5) = TP/totalP; %TP
resultados(6) = TN/totalN; %TN
resultados(7) = FP/totalN; %FP
resultados(8) = FN/totalP; %FN            