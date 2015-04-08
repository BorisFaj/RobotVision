function div = js_sample(Xp, Xq, w, numEstimate)
%Calculate Jensen-Shannon divergance JS(P||Q) from samples Xp and Xq, each example is 1-D

% JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
% M = 0.5*(P+Q)

if nargin < 3
    w = 0.05; % the kernel size (Gaussian)
    numEstimate = 100; % number of points to discretize
end

Xp = Xp(:); Xq = Xq(:);

% Calculate common supports for P and Q
minS = min([Xp;Xq],[],1);
maxS = max([Xp;Xq],[],1);

Samples = cell(1,2); % first column for Xp, second for Xq
Samples{1,1} = Xp;
Samples{1,2} = Xq;

% Estimate P and Q on the support
x_i = linspace(minS-w,maxS+w,numEstimate)';
[pxgivc, pxc, deltaX] = estimateConditionalPdf(Samples,x_i,w);

% Calculate JS(P||Q)
P = pxgivc(:,1);
Q = pxgivc(:,2);
M = 0.5*(P+Q);
div = 0.5*( kl_discrete(P,M)+kl_discrete(Q,M) );

end


