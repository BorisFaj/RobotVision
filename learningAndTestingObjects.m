function [results] = learningAndTestingObjects(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest, results)

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

if(verbose)
    fprintf('######### ######### ######### #########\n');
    fprintf('######### OBJECT CLASSIFICATION #########\n');
    fprintf('######### ######### ######### #########\n');
end

for c=1:Configuration.numObjects
    
    % We generate the training and testing sequences using +1 or -1, taking
    % into account just the current class
    
    y = clasesForTraining(c,:);
    y2 = clasesForTest(c,:);
    
    for i=1:numel(y)
        if(y(i)==0)
            y(i)=-1;
        else
            y(i)=1;
        end
    end
    
    for i=1:numel(y2)
        if(y2(i)==0)
            y2(i)=-1;
        else
            y2(i)=1;
        end
    end
    
    y_tr = y;
    y_te = y2;
    
    if(verbose)
        % train OISVM
        fprintf('Training OISVM for object %d ...\n',c);
    end
    
    %fflush(stdout);
    
    model_oisvm=k_oisvm_train(x_tr',y_tr',model_bak);
    if(verbose)
        fprintf('Testing last solution for object %d ...\n',c);
    end
    [pred_oisvm_last,~]=model_predict(x_te',model_oisvm,0);
    
    auxpred=pred_oisvm_last;
    
    for i=1:numel(pred_oisvm_last)
        if(pred_oisvm_last(i)==-1)
            auxpred(i)=0;
        else
            auxpred(i)=1;
        end
    end
    
    results (:,c+1) = auxpred';
    if(verbose)
        fprintf('Done!\n');
        fprintf('%5.2f%% of errors for this object.\n',numel(find(pred_oisvm_last~=y_te))/numel(y_te)*100);
    end
end
