function HX = entropy_sample(Xp,w,numEstimate)
% Calculate entropy HX of the sample Xp, using the pdf estimator of Xp with
% kernel size (sigma) w an number of points to be estimated numEstimate
% === Input ===
% Xp: the input sample containing examples drawn from p(x)
% w: the kernel width (sigma) of the Gaussian kernel used to estimate the
% estimator p_hat(x)
% numEstimate: the number of points needed to be calculated/discretized.
% The more, the better, but slower.
% === Output ===
% HX: the entropy H(X) of the pdf estimator p_hat(x)

if nargin < 2
    w = 0.05; % the kernel size (Gaussian)
    numEstimate = 100; % number of points to discretize
end

Xp = Xp(:); 

% Calculate support for P
minS = min(Xp,[],1);
maxS = max(Xp,[],1);

Samples = cell(1,1); % to store Xp
Samples{1,1} = Xp;

% Estimate P and Q on the support
x_i = linspace(minS-w,maxS+w,numEstimate)';
[P, pxc, deltaX] = estimateConditionalPdf(Samples,x_i,w);

% Calculate H(X)
HX = entropy_discrete(P);
% HX = HX*deltaX; % not needed

