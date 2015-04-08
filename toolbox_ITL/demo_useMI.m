% This is an easy example on how we can use mutual information as an index
% to determine if how much a feature is correlated to the class label
clear
close all
clc

%% Let's make a synthetic data set
% Generate class label
N1 = 100; N2 = 200; N3 = 100;
c = [1+zeros(N1,1); 2+zeros(N2,1); 3+zeros(N3,1)];
mu1 = 1; sigma1 = 1;
mu2 = 5; sigma2 = 1;
mu3 = 7; sigma3 = 1;
x1 = [mu1 + sigma1*randn(N1,1);
    mu2 + sigma2*randn(N2,1);
    mu3 + sigma3*randn(N3,1)];
x2 = randn(N1+N2+N3,1);
X = [x1 x2];

figure; scatter(x1,x2,20,c); daspect([1 1 1]);
xlabel('x1'); ylabel('x2');
title('data vs class scatter plot');
%% Now let's calculate the mutual information (MI) for each feature (i.e., x1 and x2)
w = 0.05; % The sigma of Gauisssian
numEstimate = 100; % number of samples when using non-parametric pdf estimation
MI = calculateMIComplete(c,X,w,numEstimate);
fprintf('the MI for features x1:%3.3f and x2:%3.3f \n',MI(1),MI(2));
