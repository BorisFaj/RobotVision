function [accuracy] = SVMHabitaciones(Configuration, featuresForTraining, featuresForTest, clasesForTraining, clasesForTest)

% All the sets are ready at this point but the multiclass problem is split
% into N binary problems using OneAgainsAll

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

dec_values = zeros(numel(clasesForTest),Configuration.numClasses);

% One versus All strategy for a multiclass classifier from N binary
% classifiers

for c=1:Configuration.numClasses
    
    % We generate the training and testing sequences using +1 or -1, taking
    % into account just the current class
    
    y = clasesForTraining;
    y2 = clasesForTest;
    
    for i=1:numel(y)
        if(y(i)==c)
            y(i)=1;
        else
            y(i)=-1;
        end
    end
    
    for i=1:numel(y2)
        if(y2(i)==c)
            y2(i)=1;
        else
            y2(i)=-1;
        end
    end
    
    y_tr = y;
    y_te = y2;
    
    % train OISVM    
    %fflush(stdout);
    
    model_oisvm=k_oisvm_train(x_tr',y_tr',model_bak);
    [pred_oisvm_last,marg]=model_predict(x_te',model_oisvm,0);
    dec_values(:,c)=marg;
end

% NORMALIZATION
normalization = true;

[m,n]=size(dec_values);

if(normalization)
    for i=1:n
        
        maxVal = max(dec_values(:,i));
        minVal = min(dec_values(:,i));
        
        for k=1:m
            
            if(dec_values(k,i)>=0)
                dec_values(k,i)=dec_values(k,i)/maxVal;
            else
                dec_values(k,i)=dec_values(k,i)/minVal*-1;
            end
        end
    end
end

[~,predict_labels]=max(dec_values,[],2);

accuracy = numel(find(predict_labels==clasesForTest))/numel(clasesForTest)*100;