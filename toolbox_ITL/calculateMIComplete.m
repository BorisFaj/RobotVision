function [MI, IGR] = calculateMIComplete(c,X,w,numEstimate)
% Calculate the MI between class-label vector c and a set of measure
% vectors X = [x1, x2, ..., xD]
%
% ===== INPUT =====
% c: N x 1 class label vector
% X: NxD matrix where D is the number of features
% w: the kernel size for Gaussian
% numEstimate: number of points to discretize
%
% ===== OUTPUT =====
% MI: 1xD row vector, each entry MI(i) is the MI of the feature i
% IGR: 1xD row vector, each entry IGR(i) is the IGR of the feature i

if nargin < 3
    w = 0.05; % the kernel size (Gaussian)
    numEstimate = 100; % number of points to discretize
end

numFeature = size(X,2);
featureList = 1:numFeature;
MI = zeros(1,numFeature);
IGR = zeros(1,numFeature);


disp('Calculating mutual information...');
tic;
for feature = featureList;
    
   
    % Make the cell array of class vs sample
    x = X(:,feature);
    [D, classOut] = makeClassSampleCellArray(x,c);
    
    % YOu can do outliers detection in each class here
    % outlier code
    
    % Estimate the pdf
    x_i = linspace(min(x(:),[],1)-w,max(x(:),[],1)+w,numEstimate)';
    [pxgivc, pxc, deltaX] = estimateConditionalPdf(D,x_i,w);
    
    
    % mutual information and IGR
    [MI(feature), IGR(feature)] = calculateMutualInformation(pxc);
    
    if mod(feature,100)==0
        disp([num2str(feature),'/',num2str(numFeature),' voxels using ',num2str(toc),' sec']);
    end
    
    
end
disp(['Calculating MI takes ',num2str(toc),' sec for ',num2str(numFeature),' voxels']);